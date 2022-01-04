import numpy as np


def jacobian(f, epsilon=1.0e-9):
    def forward_diff(X):
        dX = epsilon * np.eye(len(X))
        return np.array([f(X + dX_k) - f(X - dX_k) for dX_k in dX]).T / (2 * epsilon)

    return forward_diff
