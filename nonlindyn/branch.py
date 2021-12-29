from functools import cache
from typing import Final, Callable
import numpy as np
import itertools as it
import nonlindyn as nld
import inspect
from functools import wraps
import dataclasses
from dataclasses import dataclass


EPS: Final = np.finfo(float).eps # machine accuracy
SCALE: Final = 1.0e3  
# SCALE natural scale of functions.  
# we assume that absulate accurancy is better than SCALE*EPS

from scipy.optimize import root

def follow_path(f, x0, t, epsilon):  
    """
    f: function R^{n+1} -> R^n
    x0: starting point in R^{n+1}
    t: tangent vector in R^{n+1}
    epsilon: float, desired distance between points
    """
    xg = x0 + t/np.linalg.norm(t)*epsilon
    for i in it.count():
        def tracer(x):
            return np.append(f(x), np.linalg.norm(x-x0)-epsilon)

        sol = root(tracer, xg)  # TODO: convert to newton?
        if not sol.success:
            print(f"did not converge after {i} steps at {xg}")
            break         
        yield sol.x
        xg = 2*sol.x - x0
        x0 = sol.x

def bound_point(f, X,**kwargs):
    """
    Helper function to create `BoundPoint`. Converts `X` into numpy array
    `f` now has standard signature, and `kwargs` becomes standard parameter 
    dict `p`.
    """
    X = np.array(X)
    p = inspect.signature(f).bind_partial()
    p.apply_defaults() 
    p = p.arguments | kwargs
    
    @wraps(f)
    def fn(X, p): 
        return np.array(f(X, **p) )
    
    return BoundPoint(fn, X, p)

@dataclass
class BoundPoint:
    f: Callable[[np.ndarray, dict], np.ndarray]
    X: np.ndarray
    p: dict[str, float]
    
    def __hash__(self):
        # we need a hash function for @cache to work
        return hash((self.f, bytes(self.X.data), tuple(self.p.values())))
    
    @property
    @cache
    def fX(self):
        return self.f(self.X, self.p)

    @property
    @cache
    def DfX(self):
        f = lambda X: self.f(X, self.p)
        return nld.jacobian(f)(self.X)
    
    @cache
    def eigvals(self):
        return np.linalg.eigvals(self.DfX) 
    
    @property
    @cache
    def dim_unstable(self):
        return sum(lam.real > 0 for lam in self.eigvals())
    
    @property
    @cache
    def is_fixed_point(self):
        return np.linalg.norm(self.fX) < SCALE*EPS 
    
    @cache
    def closeby_fixed_point(self):
        f = lambda X: self.f(X, self.p)
        X = nld.newton_method(f, self.X)
        return self.__class__(self.f, X, self.p)
    
    def trajectory(self, step, raster=1):
        f = lambda X: self.f(X, self.p)
        rk4y = nld.rk4yield(f, self.X, step=step, raster=raster)
        for (t, X) in rk4y:
            yield BranchPoint(t, self.__class__(self.f, X, self.p))
    
    def follow_FP(self, parameter, delta=0.1):
        if parameter not in self.p:
            raise RuntimeError(f"Specify a valid parameter.")
        if not self.is_fixed_point:
            raise RuntimeError(f"Need to start from a fixed point.")
        f = lambda x: self.f(x[:-1], (self.p | {parameter: x[-1]}) )
        x0 = np.append(self.X, self.p[parameter])
        tangent = np.zeros_like(x0)
        tangent[-1] = np.copysign(1,delta)
        
        for x in follow_path(f, x0, tangent, epsilon=abs(delta)):
            yield self.__class__(self.f, x[:-1], (self.p |{parameter: x[-1]}))


@dataclass(order=True)
class BranchPoint:
    s: float
    bp: BoundPoint = dataclasses.field(compare=False)
    def less_than(self,t):
        return self.s<t
    def as_tuple(self):
        return (self.s, self.bp.X, self.bp.p)
    
def cut_branch(startstop, iterator):
    try:
        start, stop = startstop
        iterator = it.dropwhile(
            lambda brp: brp.less_than(start), iterator
        )
    except TypeError:
        stop = startstop
    return it.takewhile(
        lambda brp: brp.less_than(stop), iterator
    )
    
def as_tuple(iterator):
    return zip(*map(lambda brp: brp.as_tuple(), iterator))