"""SODL Semantic Cube — Python SDK wrapper for sodl-semantic-cube.

Mirrors the Rust ``LatticePoint3`` and ``SemanticCubeArtifact`` types.
Provides spatial placement of training token windows in a 3D semantic
lattice for structured batch ordering and curriculum learning.

Axes for training:
    x = topic (content hash → signed integer)
    y = complexity (lexical richness → signed integer)
    z = token density (tokens per character → signed integer)

Usage::

    from sodl_weights.semantic_cube import SemanticCube, LatticePoint3

    cube = SemanticCube()
    point = cube.place_document(text, token_count=500)
    neighbors = cube.nearest(point, k=5)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(slots=True, frozen=True)
class LatticePoint3:
    """3D signed-integer lattice point matching Rust ``LatticePoint3``."""
    x: int  # i16 range: topic
    y: int  # i16 range: complexity
    z: int  # i16 range: density
    layer: int = 0  # u8 range: hierarchy

    def manhattan_distance(self, other: "LatticePoint3") -> int:
        base = abs(self.x - other.x) + abs(self.y - other.y) + abs(self.z - other.z)
        return base if self.layer == other.layer else base + 256

    def pack(self) -> int:
        """Pack into a 64-bit integer for sorting."""
        return (
            ((self.layer & 0xFF) << 48)
            | (((self.x + 32768) & 0xFFFF) << 32)
            | (((self.y + 32768) & 0xFFFF) << 16)
            | ((self.z + 32768) & 0xFFFF)
        )


@dataclass
class CubeEntry:
    """A labeled point in the semantic cube."""
    label: str
    point: LatticePoint3


class SemanticCube:
    """Semantic cube for training document placement.

    Places documents in a 3D interpretable space:
    - **x-axis (topic)**: hash-derived topic coordinate
    - **y-axis (complexity)**: lexical richness
    - **z-axis (density)**: token-per-character density
    """

    AXES = {"x": "topic", "y": "complexity", "z": "density"}

    def __init__(self) -> None:
        self._entries: list[CubeEntry] = []

    def place_document(
        self,
        text: str,
        *,
        token_count: int | None = None,
        layer: int = 0,
    ) -> LatticePoint3:
        """Place a document in the 3D semantic lattice.

        Returns a LatticePoint3 with coordinates derived from content.
        """
        # x: topic hash → [-1000, 1000]
        h = hashlib.sha256(text[:512].encode("utf-8", errors="replace")).digest()
        x = int.from_bytes(h[:2], "big", signed=False) - 32768
        x = max(-1000, min(1000, x // 33))

        # y: complexity → [-1000, 1000]
        words = text.split()
        n_words = max(1, len(words))
        unique = len(set(w.lower() for w in words))
        avg_len = sum(len(w) for w in words) / n_words
        complexity = (unique / n_words) * min(1.0, avg_len / 10.0)
        y = int(complexity * 2000 - 1000)
        y = max(-1000, min(1000, y))

        # z: density (tokens per char) → [-1000, 1000]
        if token_count is None:
            token_count = int(n_words * 1.3)
        chars = max(1, len(text))
        density = token_count / chars  # typical: 0.2-0.5
        z = int((density - 0.3) * 5000)
        z = max(-1000, min(1000, z))

        return LatticePoint3(x=x, y=y, z=z, layer=layer)

    def add_entry(self, label: str, point: LatticePoint3) -> None:
        """Add a labeled point to the cube for kNN lookup."""
        self._entries.append(CubeEntry(label=label, point=point))

    def nearest(self, query: LatticePoint3, k: int = 5) -> list[tuple[str, int]]:
        """Find k nearest entries by Manhattan distance."""
        scored = [
            (e.label, e.point.manhattan_distance(query))
            for e in self._entries
        ]
        scored.sort(key=lambda x: x[1])
        return scored[:k]

    def sort_by_locality(
        self, items: list[tuple[int, LatticePoint3]]
    ) -> list[tuple[int, LatticePoint3]]:
        """Sort (index, point) pairs by packed lattice value for locality."""
        return sorted(items, key=lambda ip: ip[1].pack())

    def metrics(self) -> dict:
        """Return cube metrics for dashboard."""
        return {
            "cube_enabled": True,
            "cube_axes": self.AXES,
            "cube_entries": len(self._entries),
        }
