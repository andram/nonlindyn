import jax.numpy as jnp
from jax import jacfwd, jacrev, jit
import numpy as np
from collections.abc import Callable

def newtonMethod(
    f: Callable[[jnp.ndarray], jnp.ndarray],
    X0: jnp.ndarray, 
    itmax: int=10, 
    epsilon: float=1e-14
):
    X = X0.copy()
    for i in range(itmax):
        JX = jacrev(f)(X)
        fX = f(X)
        DX = jnp.linalg.solve(JX, fX)
        X -= DX
        if jnp.linalg.norm(DX) <epsilon:
            break
    else:
        raise RuntimeError(
            f"Newton's method did not converge after {itmax} steps.\n"
            f"Accuracy achieved: {jnp.linalg.norm(DX)}.\n"
            f"Desired accuracy: {epsilon}. " 
        )
    return X
