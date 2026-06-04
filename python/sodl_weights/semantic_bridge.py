import dataclasses
from typing import List, Optional

@dataclasses.dataclass
class SemanticColor32:
    hierarchy: int
    r: int
    g: int
    b: int

    def pack(self) -> int:
        return (self.hierarchy << 24) | (self.r << 16) | (self.g << 8) | self.b

    @staticmethod
    def unpack(bits: int) -> 'SemanticColor32':
        return SemanticColor32(
            hierarchy=(bits >> 24) & 0xFF,
            r=(bits >> 16) & 0xFF,
            g=(bits >> 8) & 0xFF,
            b=bits & 0xFF
        )

    def semantic_distance(self, other: 'SemanticColor32') -> int:
        hr = 0 if self.hierarchy == other.hierarchy else 1024
        dr = abs(self.r - other.r)
        dg = abs(self.g - other.g)
        db = abs(self.b - other.b)
        return hr + dr*dr + dg*dg + db*db

@dataclasses.dataclass
class LatticePoint3:
    x: int
    y: int
    z: int
    layer: int

    def manhattan_distance(self, other: 'LatticePoint3') -> int:
        base = abs(self.x - other.x) + abs(self.y - other.y) + abs(self.z - other.z)
        if self.layer == other.layer:
            return base
        else:
            return base + 256
