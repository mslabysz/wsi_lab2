import numpy as np
import matplotlib.pyplot as plt
from solver import SolverParameters, solver
from functions import quadratic
from pathlib import Path


def run_seeds(n, mu, lam, sigma, n_seeds=5):
    values = []
    for seed in range(n_seeds):
        params = SolverParameters(
            n=n, fe_max=100 * n, mu=mu, lam=lam, sigma=sigma, target=1e-4
        )
        result = solver(quadratic, params, np.random.default_rng(seed))
        values.append(result.value)
    values = np.array(values)
    return float(values.mean()), float(values.std(ddof=1))


if __name__ == "__main__":
    Path("wyniki").mkdir(exist_ok=True)
    lines = ["sigma, mu=10, lam=20", "n sigma mean std"]
    for n in (10, 30):
        for sigma in (0.1, 0.5, 2.0):
            mean, std = run_seeds(n=n, mu=10, lam=20, sigma=sigma)
            lines.append(f"{n} {sigma} {mean} {std}")
    lines += ["", "mu, lam=2*mu, sigma=0.5", "n mu lam mean std"]
    for n in (10, 30):
        for mu in (10, 20, 40):
            mean, std = run_seeds(n=n, mu=mu, lam=2 * mu, sigma=0.5)
            lines.append(f"{n} {mu} {2 * mu} {mean} {std}")
    lines += ["", "lam, mu=10, sigma=0.5", "n lam mean std"]
    for n in (10, 30):
        for lam in (5, 20, 80):
            mean, std = run_seeds(n=n, mu=10, lam=lam, sigma=0.5)
            lines.append(f"{n} {lam} {mean} {std}")
    Path("wyniki/tabela.txt").write_text("\n".join(lines) + "\n")
    print(Path("wyniki/tabela.txt").read_text())
    fig, ax = plt.subplots()
    for sigma in (0.1, 0.5, 2.0):
        params = SolverParameters(
            n=10, fe_max=100 * 10, mu=10, lam=20, sigma=sigma, target=1e-4
        )
        result = solver(quadratic, params, np.random.default_rng(0))
        ax.plot(result.history, label=f"sigma={sigma}")
    ax.set_yscale("log")
    ax.set_xlabel("pokolenie")
    ax.set_ylabel("q")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("wyniki/quadratic_n10_sigma_seed0.png", dpi=150)
