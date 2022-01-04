from functools import cache
from typing import Final, Callable
import numpy as np
import itertools as it
import nonlindyn as nld
import inspect
from functools import wraps
import dataclasses
from dataclasses import dataclass

from .boundpoint import BoundPoint
from .rk4 import rk4yield


from scipy.optimize import root


def follow_path(f, x0, t, epsilon):
    """
    f: function R^{n+1} -> R^n
    x0: starting point in R^{n+1}
    t: tangent vector in R^{n+1}
    epsilon: float, desired distance between points
    """
    xg = x0 + t / np.linalg.norm(t) * epsilon
    for i in it.count():

        def tracer(x):
            return np.append(f(x), np.linalg.norm(x - x0) - epsilon)

        sol = root(tracer, xg)  # TODO: convert to newton?
        if not sol.success:
            print(f"did not converge after {i} steps at {xg}")
            break
        yield sol.x
        xg = 2 * sol.x - x0
        x0 = sol.x


@dataclass(order=True)
class BranchPoint:
    s: float
    bp: BoundPoint = dataclasses.field(compare=False)

    def less_than(self, t):
        return self.s < t

    def bigger_than(self, t):
        return self.s > t

    def as_tuple(self):
        return (self.s, self.bp.X, self.bp.p)


def cut(startstop, iterator):
    try:
        start, stop = startstop
        iterator = it.dropwhile(
            lambda brp: brp.less_than(start) and brp.bigger_than(stop), iterator
        )
    except TypeError:
        stop = startstop
    return it.takewhile(
        lambda brp: brp.bigger_than(start) and brp.less_than(stop), iterator
    )


def as_tuple(iterator):
    return zip(*map(lambda brp: brp.as_tuple(), iterator))


def trajectory(bp, step, raster=1):
    f = lambda X: bp.f(X, bp.p)
    rk4y = rk4yield(f, bp.X, step=step, raster=raster)
    for (t, X) in rk4y:
        yield BranchPoint(t, type(bp)(bp.f, X, bp.p))


def follow_FP(bp, parameter, delta=0.1):
    if parameter not in bp.p:
        raise RuntimeError(f"Specify a valid parameter.")
    if not bp.is_fixed_point:
        raise RuntimeError(f"Need to start from a fixed point.")
    f = lambda x: bp.f(x[:-1], (bp.p | {parameter: x[-1]}))
    x0 = np.append(bp.X, bp.p[parameter])
    tangent = np.zeros_like(x0)
    tangent[-1] = np.copysign(1, delta)

    for k, x in enumerate(follow_path(f, x0, tangent, epsilon=abs(delta))):
        yield BranchPoint(
            k * delta, type(bp)(bp.f, x[:-1], (bp.p | {parameter: x[-1]}))
        )
