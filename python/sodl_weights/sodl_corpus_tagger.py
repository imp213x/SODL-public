"""SODL Corpus Tagger — Assign SemanticColor32 tags to training documents.

This module bridges SODL's Rust ``sodl-semantic-color`` crate to the
Carla training data pipeline.  Each training document receives a 32-bit
tag encoding its semantic role, topic fingerprint, complexity, and size.

Tag layout (mirrors ``SemanticColor32`` in Rust):

    ┌──────────┬──────┬──────┬──────┐
    │ hierarchy│  R   │  G   │  B   │   = 32 bits total
    │  (8 bit) │(8 b) │(8 b) │(8 b) │
    └──────────┴──────┴──────┴──────┘

    hierarchy : training‑role bucket
    R         : topic fingerprint (content hash → [0,255])
    G         : complexity score  (lexical richness → [0,255])
    B         : document length   (log₂ token count → [0,255])
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Sequence

# ---------------------------------------------------------------------------
# Role constants — used for the hierarchy byte
# ---------------------------------------------------------------------------
ROLE_TEXTBOOK = 1
ROLE_CODE = 2
ROLE_REASONING = 3
ROLE_TOOL_USE = 4
ROLE_QA = 5
ROLE_GENERIC = 0

_ROLE_MAP: dict[str, int] = {
    "textbook": ROLE_TEXTBOOK,
    "educational": ROLE_TEXTBOOK,
    "code": ROLE_CODE,
    "programming": ROLE_CODE,
    "reasoning": ROLE_REASONING,
    "math": ROLE_REASONING,
    "tool": ROLE_TOOL_USE,
    "api": ROLE_TOOL_USE,
    "qa": ROLE_QA,
    "dialogue": ROLE_QA,
}

# Fast word tokenization (letters + digits only)
_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


# ---------------------------------------------------------------------------
# SemanticColor32 — Python mirror of the Rust struct
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class SemanticColor32:
    """32‑bit semantic color tag matching ``sodl-semantic-color`` Rust crate."""

    hierarchy: int  # [0, 255]
    r: int          # [0, 255]
    g: int          # [0, 255]
    b: int          # [0, 255]

    def pack(self) -> int:
        """Pack into a 32‑bit integer in H‑R‑G‑B order."""
        return (
            ((self.hierarchy & 0xFF) << 24)
            | ((self.r & 0xFF) << 16)
            | ((self.g & 0xFF) << 8)
            | (self.b & 0xFF)
        )

    @classmethod
    def unpack(cls, bits: int) -> "SemanticColor32":
        """Unpack a 32‑bit integer into a SemanticColor32."""
        return cls(
            hierarchy=(bits >> 24) & 0xFF,
            r=(bits >> 16) & 0xFF,
            g=(bits >> 8) & 0xFF,
            b=bits & 0xFF,
        )

    def semantic_distance(self, other: "SemanticColor32") -> int:
        """Fast distance matching the Rust implementation.

        Hierarchy mismatch adds a 1024 penalty; RGB uses squared
        Euclidean distance.
        """
        hr = 0 if self.hierarchy == other.hierarchy else 1024
        dr = (self.r - other.r)
        dg = (self.g - other.g)
        db = (self.b - other.b)
        return hr + dr * dr + dg * dg + db * db

    def hue_degrees(self) -> float:
        """Compute HSL hue in [0, 360) — derived on demand (not stored)."""
        rf = self.r / 255.0
        gf = self.g / 255.0
        bf = self.b / 255.0
        mx = max(rf, gf, bf)
        mn = min(rf, gf, bf)
        delta = mx - mn
        if delta == 0.0:
            return 0.0
        if mx == rf:
            h = 60.0 * (((gf - bf) / delta) % 6.0)
        elif mx == gf:
            h = 60.0 * (((bf - rf) / delta) + 2.0)
        else:
            h = 60.0 * (((rf - gf) / delta) + 4.0)
        return h + 360.0 if h < 0.0 else h


# ---------------------------------------------------------------------------
# Document analysis helpers
# ---------------------------------------------------------------------------

def _topic_hash_byte(text: str) -> int:
    """Deterministic topic fingerprint → [0, 255].

    Uses blake3 / sha256 of the first 512 chars as a stable content hash,
    then maps the first byte to [0, 255].
    """
    sample = text[:512].encode("utf-8", errors="replace")
    digest = hashlib.blake3(sample).digest() if hasattr(hashlib, "blake3") else hashlib.sha256(sample).digest()
    return digest[0]


def _complexity_byte(text: str) -> int:
    """Lexical complexity → [0, 255].

    Combines average word length and vocabulary richness (type/token ratio).
    """
    words = _WORD_RE.findall(text)
    if not words:
        return 0
    n_words = len(words)
    n_unique = len(set(w.lower() for w in words))
    avg_word_len = sum(len(w) for w in words) / n_words  # typical: 3–8
    ttr = n_unique / n_words  # type-token ratio: 0.0–1.0

    # Combine: avg_word_len normalized to [0,1] (cap at 12) × ttr
    norm_len = min(1.0, avg_word_len / 12.0)
    score = (0.5 * norm_len + 0.5 * ttr) * 255.0
    return max(0, min(255, int(score)))


def _length_byte(token_count: int) -> int:
    """Log-bucketed document length → [0, 255]."""
    if token_count <= 0:
        return 0
    return max(0, min(255, int(math.log2(max(1, token_count)) * 16)))


def _detect_role(text: str, role_hint: str | None = None) -> int:
    """Detect training role from text content or explicit hint."""
    if role_hint:
        key = role_hint.strip().lower()
        if key in _ROLE_MAP:
            return _ROLE_MAP[key]

    # Simple heuristic detection
    sample = text[:2000].lower()

    # Code detection
    code_signals = ["def ", "class ", "import ", "function ", "return ", "const ", "var ", "pub fn "]
    code_count = sum(1 for s in code_signals if s in sample)
    if code_count >= 3:
        return ROLE_CODE

    # Math/reasoning detection
    math_signals = ["theorem", "proof", "equation", "=", "∀", "∃", "therefore", "hence"]
    math_count = sum(1 for s in math_signals if s in sample)
    if math_count >= 2:
        return ROLE_REASONING

    # QA detection
    qa_signals = ["question:", "answer:", "q:", "a:", "user:", "assistant:"]
    qa_count = sum(1 for s in qa_signals if s in sample)
    if qa_count >= 2:
        return ROLE_QA

    return ROLE_TEXTBOOK  # Default for FineWeb-Edu


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tag_document(
    text: str,
    *,
    role: str | None = None,
    token_count: int | None = None,
) -> SemanticColor32:
    """Assign a ``SemanticColor32`` tag to a training document.

    Parameters
    ----------
    text : str
        The document text.
    role : str, optional
        Explicit role hint (e.g., ``"code"``, ``"textbook"``).
        If ``None``, role is auto-detected from content.
    token_count : int, optional
        Pre-computed token count. If ``None``, estimated from word count × 1.3.
    """
    hierarchy = _detect_role(text, role)
    r = _topic_hash_byte(text)
    g = _complexity_byte(text)
    if token_count is None:
        token_count = int(len(_WORD_RE.findall(text)) * 1.3)
    b = _length_byte(token_count)
    return SemanticColor32(hierarchy=hierarchy, r=r, g=g, b=b)


def tag_documents(
    texts: Sequence[str],
    *,
    roles: Sequence[str | None] | None = None,
    token_counts: Sequence[int | None] | None = None,
) -> list[SemanticColor32]:
    """Bulk‑tag a batch of documents. Returns colors in the same order."""
    roles_iter = roles if roles is not None else [None] * len(texts)
    counts_iter = token_counts if token_counts is not None else [None] * len(texts)
    return [
        tag_document(t, role=r, token_count=c)
        for t, r, c in zip(texts, roles_iter, counts_iter)
    ]


def sort_by_color(
    items: list[tuple[int, SemanticColor32]],
) -> list[tuple[int, SemanticColor32]]:
    """Sort (index, color) pairs by packed color value for locality ordering.

    Items with the same hierarchy are grouped together, then ordered by
    RGB proximity within each group — maximizing sequential locality.
    """
    return sorted(items, key=lambda ic: ic[1].pack())
