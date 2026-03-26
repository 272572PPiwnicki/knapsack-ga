from __future__ import annotations

import random

from ga import RNG_SEED, genetic_algorithm
from knapsack import CAPACITY, VALUES, WEIGHTS, KnapsackInstance
from utils import brute_force_optimal


def main() -> None:
    inst = KnapsackInstance(tuple(WEIGHTS), tuple(VALUES), CAPACITY)
    random.seed(RNG_SEED)
    best, f = genetic_algorithm(inst)
    w = inst.total_weight(best)
    v = inst.total_value(best)
    print("Instancja: n =", len(WEIGHTS), ", W =", CAPACITY)
    print("Najlepszy osobnik (bity):", "".join(str(b) for b in best))
    print("Waga:", w, ", wartość:", v, ", fitness:", f)
    print("Dopuszczalne:", w <= CAPACITY)

    n = len(WEIGHTS)
    if n <= 22:
        opt_bits, opt_val = brute_force_optimal(WEIGHTS, VALUES, CAPACITY)
        print("Brute-force optymalna wartość:", opt_val, "| bitstring:", "".join(str(b) for b in opt_bits))
        ga_val = inst.total_value(best) if inst.total_weight(best) <= inst.capacity else None
        if ga_val is not None and ga_val == opt_val:
            print("GA trafił w optimum (wartość).")
        elif ga_val is not None:
            print("GA wartość:", ga_val, "(może być suboptymalne.")


if __name__ == "__main__":
    main()
