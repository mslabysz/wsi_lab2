from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np


from functions import f3, f18, quadratic
from solver import SolverParameters, solver

OUT = Path("wyniki")
N_SEEDS = 5
DIMENSIONS = (10, 30)
FUNCTIONS = [
    (quadratic, (-10.0, 10.0), 1e-4, (0.1, 0.5, 2.0), 0.5),
    (f3, (-100.0, 100.0), 300, (1.0, 5.0, 20.0), 5.0),
    (f18, (-100.0, 100.0), 1800, (1.0, 5.0, 20.0), 5.0),
]


def run_setting(func, n, mu, lam, sigma, bounds, target):
    values = []
    histories = []
    for seed in range(N_SEEDS):
        params = SolverParameters(
            n=n,
            fe_max=100 * n,
            mu=mu,
            lam=lam,
            sigma=sigma,
            bounds=bounds,
            target=target,
        )
        result = solver(func, params, np.random.default_rng(seed))
        values.append(result.value)
        histories.append(result.history)
    values = np.array(values)
    length = min(len(history) for history in histories)
    mean_history = np.mean([history[:length] for history in histories], axis=0)
    return float(values.mean()), float(values.std(ddof=1)), mean_history


def save_function_plot(name, panels):
    plt.rcParams["font.family"] = "DejaVu Sans"
    row_for = {"sigma": 0, "mu": 1, "lam": 2}
    col_for = {10: 0, 30: 1}
    title_for = {"sigma": "σ", "mu": "μ", "lam": "λ"}
    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    for (kind, n), curves in panels.items():
        ax = axes[row_for[kind], col_for[n]]
        for label, history in curves:
            ax.plot(history, label=label, linewidth=1)
        ax.set_yscale("log")
        ax.set_title(f"{title_for[kind]}, n = {n}")
        ax.set_xlabel("pokolenie")
        ax.set_ylabel("q(x)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{name}, średnia z {N_SEEDS} seedów")
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    lines = []
    for func, bounds, target, sigmas, sigma_ref in FUNCTIONS:
        name = func.__name__
        panels = {}
        lines += [name, "sigma, mu=10, lam=20", "n sigma mean std"]
        for n in DIMENSIONS:
            curves = []
            for sigma in sigmas:
                mean, std, history = run_setting(func, n, 10, 20, sigma, bounds, target)
                lines.append(f"{n} {sigma} {mean} {std}")
                print(name, lines[-1], flush=True)
                curves.append((f"σ = {sigma}", history))
            panels[("sigma", n)] = curves

        lines += ["", f"{name}, mu, lam=2*mu, sigma={sigma_ref}", "n mu lam mean std"]
        for n in DIMENSIONS:
            curves = []
            for mu in (10, 20, 40):
                mean, std, history = run_setting(
                    func, n, mu, 2 * mu, sigma_ref, bounds, target
                )
                lines.append(f"{n} {mu} {2 * mu} {mean} {std}")
                print(name, lines[-1], flush=True)
                curves.append((f"μ = {mu}, λ = {2 * mu}", history))
            panels[("mu", n)] = curves

        lines += ["", f"{name}, lam, mu=10, sigma={sigma_ref}", "n lam mean std"]
        for n in DIMENSIONS:
            curves = []
            for lam in (5, 20, 80):
                mean, std, history = run_setting(
                    func, n, 10, lam, sigma_ref, bounds, target
                )
                lines.append(f"{n} {lam} {mean} {std}")
                print(name, lines[-1], flush=True)
                curves.append((f"λ = {lam}", history))
            panels[("lam", n)] = curves
        save_function_plot(name, panels)
        lines.append("")

    (OUT / "tabela.txt").write_text("\n".join(lines) + "\n")
    print((OUT / "tabela.txt").read_text())
