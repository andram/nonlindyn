from functools import cache
from typing import Final
import numpy as np
import itertools as it


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

class BoundPoint:
    def __init__(self, f, X,**kwargs):
        self.f = f        # function
        self.X = X        # vector in phase space
        self.p = kwargs   # parameters

    def __repr__(self):
        return f"{self.__class__.__name__}({self.f}, {self.X}, **{self.p})"
    
    @property
    @cache
    def fX(self):
        return self.f(self.X, **self.p)

    @property
    @cache
    def DfX(self):
        f = lambda X: self.f(X, **self.p)
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
        f = lambda X: self.f(X, **self.p)
        X = nld.newton_method(f, self.X)
        return self.__class__(self.f, X, **self.p)
    
    def follow_FP(self, parameter, delta=0.1):
        if parameter not in self.p:
            raise RuntimeError(f"Specify a valid parameter.")
        if not self.is_fixed_point:
            raise RuntimeError(f"Need to start from a fixed point.")
        f = lambda x: self.f(x[:-1], **(self.p | {parameter: x[-1]}) )
        x0 = np.append(self.X, self.p[parameter])
        tangent = np.zeros_like(x0)
        tangent[-1] = np.copysign(1,delta)
        
        for x in follow_path(f, x0, tangent, epsilon=abs(delta)):
            yield self.__class__(self.f, x[:-1], **(self.p |{parameter: x[-1]}))

