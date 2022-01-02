from functools import cache
from typing import Final, Callable
import numpy as np
import itertools as it
import inspect
from functools import wraps
import dataclasses
from dataclasses import dataclass


from .jacobian import jacobian
from .newton_method import newton_method

from . import SCALE, EPS


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
        return jacobian(f)(self.X)
    
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
        X = newton_method(f, self.X)
        return self.__class__(self.f, X, self.p)
    
