from __future__ import annotations


def brute_force_optimal(weights: list[int], values: list[int], capacity: int) -> tuple[list[int], int]:
    n = len(weights)
    best_val = -1
    best_mask = 0
    for mask in range(1 << n):
        wsum = vsum = 0
        for i in range(n):
            if mask >> i & 1:
                wsum += weights[i]
                vsum += values[i]
        if wsum <= capacity and vsum > best_val:
            best_val = vsum
            best_mask = mask
    bits = [1 if (best_mask >> i) & 1 else 0 for i in range(n)]
    return bits, best_val
