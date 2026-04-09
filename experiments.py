from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ga import genetic_algorithm
from knapsack import CAPACITY, KnapsackInstance, VALUES, WEIGHTS
from utils import brute_force_optimal


N_RUNS = 30
INST = KnapsackInstance(tuple(WEIGHTS), tuple(VALUES), CAPACITY)
_, OPT_VAL = brute_force_optimal(WEIGHTS, VALUES, CAPACITY)

# defaulty ga
BASE_PARAMS: dict[str, Any] = dict(
    pop_size=80,
    generations=400,
    p_cross=0.85,
    p_mut=0.02,
    stagnation_limit=120,
)


# wynik jednego uruchomienia ga
@dataclass
class RunResult:
    value: int | None
    feasible: bool
    time_ms: float
    n_gens: int
    history: list[float]


# zagregowane wyniki z N_RUNS uruchomien
@dataclass
class AggResult:
    label: str
    mean_val: float
    std_val: float
    min_val: int
    max_val: int
    hit_rate: float
    mean_time_ms: float
    mean_gens: float
    repr_history: list[float]
    values: list[int] = field(default_factory=list)


# jedno uruchomienie
def run_once(seed: int, ga_params: dict[str, Any]) -> RunResult:
    history: list[float] = []
    t0 = time.perf_counter()
    best, _ = genetic_algorithm(INST, seed=seed, best_fitness_history=history, **ga_params)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    feasible = INST.total_weight(best) <= INST.capacity
    value = INST.total_value(best) if feasible else None
    return RunResult(
        value=value,
        feasible=feasible,
        time_ms=elapsed_ms,
        n_gens=len(history),
        history=history,
    )


# uruchomienie ga n_runs razy
def run_repeated(label: str, ga_params: dict[str, Any], n_runs: int = N_RUNS) -> AggResult:
    results = [run_once(seed, ga_params) for seed in range(n_runs)]

    vals = [r.value if r.value is not None else 0 for r in results]
    hits = sum(1 for v in vals if v == OPT_VAL)

    return AggResult(
        label=label,
        mean_val=statistics.mean(vals),
        std_val=statistics.stdev(vals) if len(vals) > 1 else 0.0,
        min_val=min(vals),
        max_val=max(vals),
        hit_rate=hits / n_runs,
        mean_time_ms=statistics.mean(r.time_ms for r in results),
        mean_gens=statistics.mean(r.n_gens for r in results),
        repr_history=results[0].history,
        values=vals,
    )


def experiment_sweep(
    param_name: str,
    values: list[Any],
    base_params: dict[str, Any] = BASE_PARAMS,
) -> list[AggResult]:
    agg_results = []
    for v in values:
        params = {**base_params, param_name: v}
        label = f"{param_name}={v}"
        agg = run_repeated(label, params)
        agg_results.append(agg)
    return agg_results


# tabela wynikow
def print_table(title: str, rows: list[AggResult]) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}   (optimum = {OPT_VAL},  N={N_RUNS})")
    print(f"{'='*70}")
    header = (
        f"{'Wariant':<22} {'Śr.wartość':>11} {'Std':>7} "
        f"{'Min':>5} {'Max':>5} {'Hit%':>6} {'Czas[ms]':>10} {'Śr.gen':>8}"
    )
    print(header)
    print("-" * 70)
    for r in rows:
        print(
            f"{r.label:<22} "
            f"{r.mean_val:>11.2f} "
            f"{r.std_val:>7.2f} "
            f"{r.min_val:>5} "
            f"{r.max_val:>5} "
            f"{r.hit_rate*100:>5.1f}% "
            f"{r.mean_time_ms:>10.2f} "
            f"{r.mean_gens:>8.1f}"
        )
    print("=" * 70)


# wykres zbieznosci
def plot_convergence(title: str, rows: list[AggResult], filename: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in rows:
        ax.plot(r.repr_history, label=r.label)
    ax.axhline(OPT_VAL, color="black", linestyle="--", linewidth=1.2, label=f"optimum={OPT_VAL}")
    ax.set_title(title)
    ax.set_xlabel("Generacja")
    ax.set_ylabel("Najlepszy fitness (seed=0)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    print(f"  -> wykres zapisany: {filename}")


def plot_boxplot(title: str, rows: list[AggResult], filename: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [r.values for r in rows]
    labels = [r.label for r in rows]
    ax.boxplot(data, tick_labels=labels, patch_artist=True)
    ax.axhline(OPT_VAL, color="red", linestyle="--", linewidth=1.2, label=f"optimum={OPT_VAL}")
    ax.set_title(title)
    ax.set_xlabel("Wariant")
    ax.set_ylabel("Wartość plecaka")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    print(f"  -> boxplot zapisany: {filename}")


def main() -> None:
    print(f"Optimum (brute-force): {OPT_VAL}")
    print(f"Liczba powtórzeń na wariant: {N_RUNS}\n")

    # E1 baseline
    print("Uruchamiam E1 baseline...")
    e1 = [run_repeated("baseline", BASE_PARAMS)]
    print_table("E1 Baseline (domyślne parametry)", e1)

    # E2 pop_size
    print("\nUruchamiam E2 pop_size...")
    e2 = experiment_sweep("pop_size", [20, 40, 80, 120, 200])
    print_table("E2 Wpływ pop_size", e2)
    plot_convergence("E2 Zbieżność dla różnych pop_size", e2, "e2_convergence.png")
    plot_boxplot("E2 Rozkład wartości dla różnych pop_size", e2, "e2_boxplot.png")

    # E3 p_mut
    print("\nUruchamiam E3 p_mut...")
    e3 = experiment_sweep("p_mut", [0.005, 0.01, 0.02, 0.05, 0.10])
    print_table("E3 Wpływ p_mut", e3)
    plot_convergence("E3 Zbieżność dla różnych p_mut", e3, "e3_convergence.png")
    plot_boxplot("E3 Rozkład wartości dla różnych p_mut", e3, "e3_boxplot.png")

    # E4 p_cross
    print("\nUruchamiam E4 p_cross...")
    e4 = experiment_sweep("p_cross", [0.5, 0.7, 0.85, 0.95, 1.0])
    print_table("E4 Wpływ p_cross", e4)
    plot_convergence("E4 Zbieżność dla różnych p_cross", e4, "e4_convergence.png")
    plot_boxplot("E4 Rozkład wartości dla różnych p_cross", e4, "e4_boxplot.png")

    # E5 stagnation_limit
    print("\nUruchamiam E5 stagnation_limit...")
    e5 = experiment_sweep("stagnation_limit", [50, 100, 120, 200, None])
    print_table("E5 Wpływ stagnation_limit", e5)
    plot_convergence("E5 Zbieżność dla różnych stagnation_limit", e5, "e5_convergence.png")
    plot_boxplot("E5 Rozkład wartości dla różnych stagnation_limit", e5, "e5_boxplot.png")

    print("\nGotowe. Wykresy zapisane w bieżącym katalogu.")


if __name__ == "__main__":
    main()
