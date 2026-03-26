from __future__ import annotations

from dataclasses import dataclass

WEIGHTS = [2, 3, 4, 5, 9, 7, 8, 6]
VALUES = [6, 5, 8, 9, 12, 10, 11, 7]
CAPACITY = 20


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
