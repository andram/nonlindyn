import unittest
import numpy as np
import nonlindyn as nld

def Lorenz(
        X:  np.ndarray,
        sigma:float=10.,
        beta:float=8/3.,
        rho:float=28.
) -> np.ndarray:
    x,y,z = X
    dx = sigma * (y - x)
    dy = x * (rho -z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])

X0 = np.array([1.,2.,3.])


class TestRK4Methods(unittest.TestCase):

    def test_Lorenz(self):
        self.assertAlmostEqual( Lorenz(X0)[0], 10.)
        self.assertAlmostEqual( Lorenz(X0)[1], 23.)
        self.assertAlmostEqual( Lorenz(X0)[2], -6.0)

    def test_Lorenz2(self):
        self.assertTrue( all(np.isclose(Lorenz(X0), [10.,23.,-6.])))

    def test_rk4(self):
        self.assertTrue( all( np.isclose(
            nld.rk4trajectory(Lorenz,X0,stop=2.,step=0.0001)[-1][-1],
            [ -7.97525832, -10.0723744 ,  23.19144004]
        )))

if __name__ == '__main__':
    unittest.main()
