from __future__ import annotations

import random
from typing import Callable

from knapsack import KnapsackInstance

POPULATION_SIZE = 80
P_CROSS = 0.85
P_MUT = 0.02
GENERATIONS = 400
STAGNATION_LIMIT: int | None = 120
TOURNAMENT_K = 3
ELITE_COUNT = 2
RNG_SEED = 42


def random_individual(n: int) -> list[int]:
    return [random.randint(0, 1) for _ in range(n)]


def tournament_select(pop: list[list[int]], k: int, fit: Callable[[list[int]], float]) -> list[int]:
    candidates = random.sample(pop, k=min(k, len(pop)))
    return max(candidates, key=fit)


def crossover_one_point(a: list[int], b: list[int]) -> tuple[list[int], list[int]]:
    n = len(a)
    if n < 2:
        return a[:], b[:]
    point = random.randint(1, n - 1)
    c1 = a[:point] + b[point:]
    c2 = b[:point] + a[point:]
    return c1, c2


def mutate(bits: list[int], p_m: float) -> None:
    for i in range(len(bits)):
        if random.random() < p_m:
            bits[i] = 1 - bits[i]


def genetic_algorithm(
    inst: KnapsackInstance,
    *,
    pop_size: int = POPULATION_SIZE,
    generations: int = GENERATIONS,
    p_cross: float = P_CROSS,
    p_mut: float = P_MUT,
    stagnation_limit: int | None = STAGNATION_LIMIT,
    seed: int | None = RNG_SEED,
    best_fitness_history: list[float] | None = None,
) -> tuple[list[int], float]:
    fit = inst.fitness
    if seed is not None:
        random.seed(seed)

    n = len(inst.weights)
    population = [random_individual(n) for _ in range(pop_size)]
    best = max(population, key=fit)
    best_fit = fit(best)
    stagnant = 0

    for _ in range(generations):
        population.sort(key=fit, reverse=True)
        if fit(population[0]) > best_fit:
            best_fit = fit(population[0])
            best = population[0][:]
            stagnant = 0
        else:
            stagnant += 1
        if best_fitness_history is not None:
            best_fitness_history.append(best_fit)
        if stagnation_limit is not None and stagnant >= stagnation_limit:
            break

        new_pop: list[list[int]] = []
        new_pop.extend(population[i][:] for i in range(min(ELITE_COUNT, len(population))))

        while len(new_pop) < pop_size:
            p1 = tournament_select(population, TOURNAMENT_K, fit)
            p2 = tournament_select(population, TOURNAMENT_K, fit)
            if random.random() < p_cross:
                c1, c2 = crossover_one_point(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]
            mutate(c1, p_mut)
            mutate(c2, p_mut)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        population = new_pop

    population.sort(key=fit, reverse=True)
    if fit(population[0]) > best_fit:
        best = population[0][:]
        best_fit = fit(best)
    return best, best_fit
