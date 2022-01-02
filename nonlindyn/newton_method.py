import numpy as np
from collections.abc import Callable

from .jacobian import jacobian


def newton_method(
    f: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray, 
    itmax: int=10, 
    epsilon: float=1e-14,
    jac = None
):
    jac = jacobian(f) if jac is None else jac
    for i in range(itmax):
        DX = np.linalg.solve(jac(X), f(X))
        X = X - DX
        if np.linalg.norm(DX) <epsilon: break
    else:
        raise RuntimeError(
            f"Newton's method did not converge after {itmax} steps.\n"
            f"Accuracy achieved: {np.linalg.norm(DX)}.\n"
            f"Desired accuracy: {epsilon}. " 
        )
    return X
