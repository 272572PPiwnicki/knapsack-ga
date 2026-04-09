from __future__ import annotations

from dataclasses import dataclass

WEIGHTS = [7, 3, 10, 4, 9, 2, 8, 5, 6, 11, 3, 7, 4, 9, 6, 2, 8, 5]
VALUES  = [13, 5, 18, 7, 16, 3, 14, 9, 11, 20, 6, 12, 8, 17, 10, 4, 15, 8]
CAPACITY = 45


@dataclass(frozen=True)
class KnapsackInstance:
    weights: tuple[int, ...]
    values: tuple[int, ...]
    capacity: int

    def total_weight(self, bits: list[int]) -> int:
        return sum(self.weights[i] for i, b in enumerate(bits) if b)

    def total_value(self, bits: list[int]) -> int:
        return sum(self.values[i] for i, b in enumerate(bits) if b)

    def fitness(self, bits: list[int]) -> float:
        w = self.total_weight(bits)
        v = self.total_value(bits)
        if w <= self.capacity:
            return float(v)
        over = w - self.capacity
        return float(v) - 1000.0 * over
