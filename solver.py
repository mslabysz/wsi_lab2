from dataclasses import dataclass
import numpy as np


@dataclass
class SolverParameters:
    mu: int = 10
    lam: int = 20
    sigma: float = 0.5
    bounds: tuple[float, float] = (-10.0, 10.0)
    fe_max: int = 1000
    n: int = 1
    target: float | None = None


@dataclass
class SolverResult:
    x: np.ndarray
    history: list[float]
    stop_reason: str
    fe: int
    value: float


def solver(eval_func, params, rng) -> SolverResult:
    low, high = params.bounds
    x = rng.uniform(low, high, size=(params.mu, params.n))
    values = np.array([eval_func(parent) for parent in x])
    fe = params.mu
    history = [float(values.min())]
    stop_reason = "fe_max"
    while fe < params.fe_max:
        if params.target is not None and values.min() <= params.target:
            stop_reason = "target"
            break
        n_children = min(params.lam, params.fe_max - fe)
        parents = rng.integers(0, params.mu, size=n_children)
        children = x[parents] + params.sigma * rng.normal(size=(n_children, params.n))
        children = np.clip(children, low, high)
        child_values = np.array([eval_func(child) for child in children])
        fe += n_children
        pool_x = np.vstack([x, children])
        pool_values = np.concatenate([values, child_values])
        best = np.argsort(pool_values)[: params.mu]
        x = pool_x[best]
        values = pool_values[best]
        history.append(float(values.min()))
    i = int(np.argmin(values))
    return SolverResult(
        x=x[i], value=float(values[i]), history=history, stop_reason=stop_reason, fe=fe
    )