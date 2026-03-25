"""Shared utilities for the financial reconciliation system."""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class MatchResult:
    """Represents a single transaction match between bank and check register."""
    bank_id: str
    register_id: str
    confidence: float
    match_phase: str
    flags: List[str] = field(default_factory=list)

    @property
    def bank_num(self) -> int:
        """Extract numeric ID from bank transaction ID (e.g., 'B0047' -> 47)."""
        return int(re.sub(r'[^\d]', '', self.bank_id))

    @property
    def register_num(self) -> int:
        """Extract numeric ID from register transaction ID (e.g., 'R0047' -> 47)."""
        return int(re.sub(r'[^\d]', '', self.register_id))

    @property
    def is_correct(self) -> bool:
        """Check if match is correct using ground truth (B0047 <-> R0047)."""
        return self.bank_num == self.register_num


def extract_id_number(transaction_id: str) -> int:
    """Extract numeric part from a transaction ID string."""
    return int(re.sub(r'[^\d]', '', transaction_id))


def normalize_text(text: str) -> str:
    """Basic text normalization: lowercase, strip extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'#\d+', '', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


MERCHANT_ALIASES = {
    'aazon.com': 'amazon',
    'amazon.com': 'amazon',
    'bp gas': 'gas station',
    'chevron': 'gas station',
    'exxon': 'gas station',
    'shell': 'gas station',
    'kroger': 'grocery',
    'safeway': 'grocery',
    'trader joes': 'grocery',
    'whole foods': 'grocery',
    'netflix': 'subscription',
    'spotify': 'subscription',
    'gym membership': 'subscription',
    'online pmt water': 'utility',
    'online pmt elec co': 'utility',
    'online pmt gas co': 'utility',
    'online pmt gas o': 'utility',
    'health ins pmt': 'insurance',
    'auto ins': 'insurance',
    'insurance pmt': 'insurance',
    'atm wd': 'atm withdrawal',
    'atm withdrawal': 'atm withdrawal',
    'atm cash': 'atm withdrawal',
    'direct dep payroll': 'payroll',
    'direct deposit': 'payroll',
    'ach credit salary': 'payroll',
    'ach transfer': 'transfer',
    'online transfer': 'transfer',
    'xfer to savings': 'transfer',
    'check': 'check payment',
    'ck': 'check payment',
    'monthly fee': 'bank fee',
    'maint fee': 'bank fee',
    'service charge': 'bank fee',
    'cafe': 'restaurant',
    'diner': 'restaurant',
    'bistro': 'restaurant',
    'restaurant': 'restaurant',
    'online order': 'online purchase',
    'ebay purchase': 'online purchase',
}


def categorize_description(text: str) -> str:
    """Map a normalized description to a high-level category."""
    text_lower = normalize_text(text)

    for pattern, category in MERCHANT_ALIASES.items():
        if pattern in text_lower:
            return category

    if 'misc transaction' in text_lower:
        return 'miscellaneous'
    if any(kw in text_lower for kw in ['online', 'amazon', 'ebay', 'order']):
        return 'online purchase'
    if any(kw in text_lower for kw in ['gas', 'fuel', 'fill up']):
        return 'gas station'
    if any(kw in text_lower for kw in ['grocer', 'food', 'market']):
        return 'grocery'
    if any(kw in text_lower for kw in ['restaurant', 'dinner', 'lunch', 'eating']):
        return 'restaurant'

    return 'other'
