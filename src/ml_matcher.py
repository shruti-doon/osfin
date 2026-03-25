"""Phase 2: ML-based matching using hybrid SVD + sentence embeddings.

Implements the paper's cross-language information retrieval approach
combined with modern sentence embeddings for richer text understanding.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize as sklearn_normalize
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from src.utils import MatchResult
from src.preprocessor import normalize_description, extract_text_tokens


class HybridMatcher:
    """Hybrid SVD + embedding matcher for financial transaction reconciliation.

    Implements a multi-step approach:
    1. Build parallel corpus from seed matches (unique-amount phase)
    2. Compute term alignments via pointwise mutual information (PMI)
    3. Build feature vectors combining text + numerical features
    4. Apply SVD dimensionality reduction
    5. Match via cosine similarity with Hungarian algorithm
    6. Iteratively expand the corpus with high-confidence matches
    """

    def __init__(
        self,
        svd_components: int = 40,
        svd_weight: float = 0.4,
        embedding_weight: float = 0.3,
        numerical_weight: float = 0.3,
        high_confidence_threshold: float = 0.85,
        max_iterations: int = 3,
        use_sentence_transformers: bool = True,
    ):
        """Initialize the hybrid matcher.

        Args:
            svd_components: Number of SVD components (paper suggests ~40)
            svd_weight: Weight for SVD similarity in ensemble
            embedding_weight: Weight for embedding similarity in ensemble
            numerical_weight: Weight for numerical feature similarity
            high_confidence_threshold: Threshold for adding ML matches to corpus
            max_iterations: Maximum number of iterative refinement rounds
            use_sentence_transformers: Whether to use sentence-transformers embeddings
        """
        self.svd_components = svd_components
        self.svd_weight = svd_weight
        self.embedding_weight = embedding_weight
        self.numerical_weight = numerical_weight
        self.high_confidence_threshold = high_confidence_threshold
        self.max_iterations = max_iterations
        self.use_sentence_transformers = use_sentence_transformers

        self._embedder = None
        self._svd = None
        self._bank_vocab = {}
        self._reg_vocab = {}
        self._alignment_matrix = None

    def _get_embedder(self):
        """Lazy-load sentence transformer model."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                self.use_sentence_transformers = False
                self._embedder = None
        return self._embedder

    def _build_vocabulary(
        self,
        bank_df: pd.DataFrame,
        register_df: pd.DataFrame,
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Build vocabulary mappings for bank and register descriptions."""
        bank_vocab = {}
        reg_vocab = {}

        for desc in bank_df['description']:
            for token in extract_text_tokens(desc):
                if token not in bank_vocab:
                    bank_vocab[token] = len(bank_vocab)

        for desc in register_df['description']:
            for token in extract_text_tokens(desc):
                if token not in reg_vocab:
                    reg_vocab[token] = len(reg_vocab)

        return bank_vocab, reg_vocab

    def _compute_pmi_alignment(
        self,
        seed_matches: List[MatchResult],
        bank_df: pd.DataFrame,
        register_df: pd.DataFrame,
    ) -> np.ndarray:
        """Compute pointwise mutual information alignment matrix.

        For each matched pair, record co-occurrences of bank description
        tokens with register description tokens. Then compute PMI.

        Returns:
            Alignment matrix of shape (|bank_vocab|, |reg_vocab|)
        """
        bank_lookup = bank_df.set_index('transaction_id')
        reg_lookup = register_df.set_index('transaction_id')

        n_bank = len(self._bank_vocab)
        n_reg = len(self._reg_vocab)

        if n_bank == 0 or n_reg == 0:
            return np.zeros((max(n_bank, 1), max(n_reg, 1)))

        # Co-occurrence counts
        cooccur = np.zeros((n_bank, n_reg))
        bank_counts = np.zeros(n_bank)
        reg_counts = np.zeros(n_reg)

        n_pairs = len(seed_matches)
        if n_pairs == 0:
            return np.zeros((n_bank, n_reg))

        for match in seed_matches:
            if match.bank_id not in bank_lookup.index or match.register_id not in reg_lookup.index:
                continue

            bank_desc = bank_lookup.loc[match.bank_id, 'description']
            reg_desc = reg_lookup.loc[match.register_id, 'description']

            bank_tokens = extract_text_tokens(bank_desc)
            reg_tokens = extract_text_tokens(reg_desc)

            for bt in bank_tokens:
                if bt in self._bank_vocab:
                    bi = self._bank_vocab[bt]
                    bank_counts[bi] += 1
                    for rt in reg_tokens:
                        if rt in self._reg_vocab:
                            ri = self._reg_vocab[rt]
                            cooccur[bi, ri] += 1

            for rt in reg_tokens:
                if rt in self._reg_vocab:
                    ri = self._reg_vocab[rt]
                    reg_counts[ri] += 1

        # Compute PMI
        pmi = np.zeros((n_bank, n_reg))
        for bi in range(n_bank):
            for ri in range(n_reg):
                if cooccur[bi, ri] > 0 and bank_counts[bi] > 0 and reg_counts[ri] > 0:
                    p_joint = cooccur[bi, ri] / n_pairs
                    p_bank = bank_counts[bi] / n_pairs
                    p_reg = reg_counts[ri] / n_pairs
                    pmi_val = np.log2(p_joint / (p_bank * p_reg + 1e-10) + 1e-10)
                    pmi[bi, ri] = max(pmi_val, 0)  # Positive PMI only

        return pmi

    def _build_term_vectors(
        self,
        df: pd.DataFrame,
        vocab: Dict[str, int],
        is_bank: bool,
    ) -> np.ndarray:
        """Build term-frequency vectors for transactions.

        Returns:
            Matrix of shape (n_transactions, vocab_size)
        """
        n = len(df)
        v = len(vocab)
        if v == 0:
            return np.zeros((n, 1))

        matrix = np.zeros((n, v))

        for i, (_, row) in enumerate(df.iterrows()):
            tokens = extract_text_tokens(row['description'])
            for token in tokens:
                if token in vocab:
                    matrix[i, vocab[token]] = 1.0

        # L2 normalize
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        matrix = matrix / norms

        return matrix

    def _compute_embedding_features(
        self,
        descriptions: List[str],
        vectorizer=None,
    ) -> np.ndarray:
        """Compute sentence embeddings for descriptions.

        Args:
            descriptions: List of description strings
            vectorizer: Pre-fitted TfidfVectorizer (for fallback mode consistency)

        Returns:
            Matrix of shape (n_descriptions, embedding_dim)
        """
        if self.use_sentence_transformers:
            embedder = self._get_embedder()
            if embedder is not None:
                normalized = [normalize_description(d) for d in descriptions]
                embeddings = embedder.encode(normalized, show_progress_bar=False)
                return embeddings

        # Fallback: simple character n-gram approach
        from sklearn.feature_extraction.text import TfidfVectorizer
        normalized = [normalize_description(d) for d in descriptions]
        if vectorizer is not None:
            result = vectorizer.transform(normalized).toarray()
        else:
            vectorizer = TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(2, 4),
                max_features=200,
            )
            result = vectorizer.fit_transform(normalized).toarray()
        return result

    def _compute_numerical_features(self, df: pd.DataFrame) -> np.ndarray:
        """Build numerical feature vectors.

        Features: amount_log, day_sin, day_cos, type_num
        """
        features = df[['amount_log', 'day_sin', 'day_cos', 'type_num']].values.astype(float)

        # Normalize each feature to [0, 1]
        for col in range(features.shape[1]):
            col_min = features[:, col].min()
            col_max = features[:, col].max()
            if col_max > col_min:
                features[:, col] = (features[:, col] - col_min) / (col_max - col_min)

        return features

    def _compute_similarity_matrix(
        self,
        bank_df: pd.DataFrame,
        register_df: pd.DataFrame,
        seed_matches: List[MatchResult],
    ) -> np.ndarray:
        """Compute the full similarity matrix between bank and register transactions.

        Combines:
        1. SVD-projected term similarity (paper's approach)
        2. Sentence embedding similarity (modern NLP)
        3. Numerical feature similarity

        Returns:
            Similarity matrix of shape (n_bank, n_register)
        """
        n_bank = len(bank_df)
        n_reg = len(register_df)

        # --- 1. SVD similarity ---
        # Build term vectors
        bank_terms = self._build_term_vectors(bank_df, self._bank_vocab, is_bank=True)
        reg_terms = self._build_term_vectors(register_df, self._reg_vocab, is_bank=False)

        # Apply alignment: project register terms into bank term space
        if self._alignment_matrix is not None and self._alignment_matrix.shape[0] > 0:
            # Transform: bank_terms stays as-is, reg_terms projected through alignment
            # alignment is (bank_vocab, reg_vocab), reg_terms is (n_reg, reg_vocab)
            # projected = reg_terms @ alignment.T = (n_reg, bank_vocab)
            reg_projected = reg_terms @ self._alignment_matrix.T

            # Combine
            combined_bank = bank_terms
            combined_reg = reg_projected

            # SVD reduction
            n_components = min(self.svd_components, combined_bank.shape[1] - 1, combined_reg.shape[1] - 1)
            if n_components > 0:
                # Stack and fit SVD on all data
                all_data = np.vstack([combined_bank, combined_reg])
                n_components = min(n_components, all_data.shape[1] - 1, all_data.shape[0] - 1)
                if n_components > 0:
                    self._svd = TruncatedSVD(n_components=n_components, random_state=42)
                    all_projected = self._svd.fit_transform(all_data)
                    bank_svd = all_projected[:n_bank]
                    reg_svd = all_projected[n_bank:]

                    # Cosine similarity
                    svd_sim = 1 - cdist(bank_svd, reg_svd, metric='cosine')
                    svd_sim = np.nan_to_num(svd_sim, nan=0.0)
                else:
                    svd_sim = np.zeros((n_bank, n_reg))
            else:
                svd_sim = np.zeros((n_bank, n_reg))
        else:
            svd_sim = np.zeros((n_bank, n_reg))

        # --- 2. Embedding similarity ---
        bank_descs = bank_df['description'].tolist()
        reg_descs = register_df['description'].tolist()

        if self.use_sentence_transformers:
            bank_emb = self._compute_embedding_features(bank_descs)
            reg_emb = self._compute_embedding_features(reg_descs)
        else:
            # TF-IDF fallback: fit on combined corpus for consistent dimensionality
            from sklearn.feature_extraction.text import TfidfVectorizer
            all_descs = [normalize_description(d) for d in bank_descs + reg_descs]
            vectorizer = TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=(2, 4),
                max_features=200,
            )
            vectorizer.fit(all_descs)
            bank_emb = self._compute_embedding_features(bank_descs, vectorizer=vectorizer)
            reg_emb = self._compute_embedding_features(reg_descs, vectorizer=vectorizer)

        emb_sim = 1 - cdist(bank_emb, reg_emb, metric='cosine')
        emb_sim = np.nan_to_num(emb_sim, nan=0.0)

        # --- 3. Numerical similarity ---
        bank_num = self._compute_numerical_features(bank_df)
        reg_num = self._compute_numerical_features(register_df)

        num_sim = 1 - cdist(bank_num, reg_num, metric='euclidean')
        # Normalize to [0, 1]
        num_min = num_sim.min()
        num_max = num_sim.max()
        if num_max > num_min:
            num_sim = (num_sim - num_min) / (num_max - num_min)
        else:
            num_sim = np.zeros_like(num_sim)

        # --- 4. Amount match bonus ---
        # Strong boost for same-amount pairs
        bank_amounts = bank_df['amount'].values.reshape(-1, 1)
        reg_amounts = register_df['amount'].values.reshape(1, -1)
        amount_match = (np.abs(bank_amounts - reg_amounts) < 0.01).astype(float)

        # --- Ensemble ---
        similarity = (
            self.svd_weight * svd_sim +
            self.embedding_weight * emb_sim +
            self.numerical_weight * num_sim +
            0.3 * amount_match  # bonus for exact amount matches
        )

        # Normalize to [0, 1]
        sim_min = similarity.min()
        sim_max = similarity.max()
        if sim_max > sim_min:
            similarity = (similarity - sim_min) / (sim_max - sim_min)

        return similarity

    def match(
        self,
        bank_df: pd.DataFrame,
        register_df: pd.DataFrame,
        seed_matches: List[MatchResult],
        matched_bank_ids: Optional[Set[str]] = None,
        matched_reg_ids: Optional[Set[str]] = None,
    ) -> List[MatchResult]:
        """Run the full hybrid matching pipeline.

        Args:
            bank_df: Preprocessed bank statements
            register_df: Preprocessed check register
            seed_matches: Matches from unique-amount phase (parallel corpus)
            matched_bank_ids: Already matched bank IDs to exclude
            matched_reg_ids: Already matched register IDs to exclude

        Returns:
            List of MatchResult for ML-phase matches
        """
        if matched_bank_ids is None:
            matched_bank_ids = {m.bank_id for m in seed_matches}
        if matched_reg_ids is None:
            matched_reg_ids = {m.register_id for m in seed_matches}

        all_ml_matches = []
        current_seeds = list(seed_matches)
        current_matched_bank = set(matched_bank_ids)
        current_matched_reg = set(matched_reg_ids)

        for iteration in range(self.max_iterations):
            # Filter to unmatched transactions
            bank_remaining = bank_df[~bank_df['transaction_id'].isin(current_matched_bank)].reset_index(drop=True)
            reg_remaining = register_df[~register_df['transaction_id'].isin(current_matched_reg)].reset_index(drop=True)

            if len(bank_remaining) == 0 or len(reg_remaining) == 0:
                break

            # Build vocabulary from ALL data (to capture global patterns)
            self._bank_vocab, self._reg_vocab = self._build_vocabulary(bank_df, register_df)

            # Compute PMI alignment from current seed matches
            self._alignment_matrix = self._compute_pmi_alignment(
                current_seeds, bank_df, register_df
            )

            # Compute similarity matrix for remaining transactions
            similarity = self._compute_similarity_matrix(
                bank_remaining, reg_remaining, current_seeds
            )

            # Hungarian algorithm for optimal 1-to-1 matching
            n_bank = len(bank_remaining)
            n_reg = len(reg_remaining)

            # Convert similarity to cost (for minimization)
            cost_matrix = 1.0 - similarity

            # Handle non-square matrices
            if n_bank <= n_reg:
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
            else:
                col_ind, row_ind = linear_sum_assignment(cost_matrix.T)

            # Create match results for this iteration
            iteration_matches = []
            for r, c in zip(row_ind, col_ind):
                if r < n_bank and c < n_reg:
                    conf = similarity[r, c]
                    bank_id = bank_remaining.iloc[r]['transaction_id']
                    reg_id = reg_remaining.iloc[c]['transaction_id']

                    flags = [f'iteration={iteration + 1}']
                    if conf < 0.5:
                        flags.append('low_confidence')

                    match = MatchResult(
                        bank_id=bank_id,
                        register_id=reg_id,
                        confidence=round(float(conf), 4),
                        match_phase='ml',
                        flags=flags,
                    )
                    iteration_matches.append(match)

            # Separate high and low confidence matches
            high_conf = [m for m in iteration_matches if m.confidence >= self.high_confidence_threshold]
            low_conf = [m for m in iteration_matches if m.confidence < self.high_confidence_threshold]

            # Add high-confidence matches to the seed corpus
            for m in high_conf:
                current_matched_bank.add(m.bank_id)
                current_matched_reg.add(m.register_id)
                current_seeds.append(m)

            all_ml_matches.extend(high_conf)

            # If no high-confidence matches found, add all remaining and stop
            if not high_conf:
                all_ml_matches.extend(low_conf)
                break

        # Final pass: match any remaining transactions
        final_bank_remaining = bank_df[~bank_df['transaction_id'].isin(current_matched_bank)].reset_index(drop=True)
        final_reg_remaining = register_df[~register_df['transaction_id'].isin(current_matched_reg)].reset_index(drop=True)

        if len(final_bank_remaining) > 0 and len(final_reg_remaining) > 0:
            # Rebuild with expanded corpus
            self._bank_vocab, self._reg_vocab = self._build_vocabulary(bank_df, register_df)
            self._alignment_matrix = self._compute_pmi_alignment(
                current_seeds, bank_df, register_df
            )
            similarity = self._compute_similarity_matrix(
                final_bank_remaining, final_reg_remaining, current_seeds
            )

            n_bank = len(final_bank_remaining)
            n_reg = len(final_reg_remaining)
            cost_matrix = 1.0 - similarity

            if n_bank <= n_reg:
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
            else:
                col_ind, row_ind = linear_sum_assignment(cost_matrix.T)

            for r, c in zip(row_ind, col_ind):
                if r < n_bank and c < n_reg:
                    conf = similarity[r, c]
                    bank_id = final_bank_remaining.iloc[r]['transaction_id']
                    reg_id = final_reg_remaining.iloc[c]['transaction_id']

                    match = MatchResult(
                        bank_id=bank_id,
                        register_id=reg_id,
                        confidence=round(float(conf), 4),
                        match_phase='ml',
                        flags=[f'final_pass'],
                    )
                    all_ml_matches.append(match)

        return all_ml_matches
