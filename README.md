# nonlindyn

Library to support analysing nonlinear dynamical systems

## Basic ideas

a dynamical system is described by the following

- a set of dynamical equations
- a set of constraints
- a set of parameters

The distinction between parameters and constraints is just technical. Parameters are meant to be changed in the analysis, while constraints remain fix. 

There are three types of symbols to make up those equations

- dynamical variables $x_0,\ldots,x_{n-1}$
- derivatives of dynamical variables  $\dot{x}_0,\ldots,\dot{x}_{n-1}$
- parameters $p_0, \ldots, p_{m-1}$


Dynamical equations are a list of equations which contain derivatives and possibly t.

Special case happens, when we find solutions of all equations with $\dot{x}_k=0$ independent of t. Such solutions are called fixed points. 

Stability of FPs given by eigenvalues of Jacobian.

Now consider what happens at a Hopf bifurcation. 
