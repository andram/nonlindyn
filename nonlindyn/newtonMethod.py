import numpy as np
from collections.abc import Callable


def jacobian(f, epsilon = 1.0e-9):
    def forward_diff(X):
        dX = epsilon * np.eye(len(X))
        return np.array(
            [ f(X+dX_k) - f(X-dX_k) for dX_k in dX]
        ).T/(2*epsilon)
    return forward_diff


def newton_method(
    f: Callable[[np.ndarray], np.ndarray],
    X0: np.ndarray, 
    itmax: int=10, 
    epsilon: float=1e-14
):
    X = X0.copy()
    for i in range(itmax):
        JX = jacobian(f)(X)
        fX = f(X)
        DX = np.linalg.solve(JX, fX)
        X -= DX
        if np.linalg.norm(DX) <epsilon:
            break
    else:
        raise RuntimeError(
            f"Newton's method did not converge after {itmax} steps.\n"
            f"Accuracy achieved: {np.linalg.norm(DX)}.\n"
            f"Desired accuracy: {epsilon}. " 
        )
    return X
