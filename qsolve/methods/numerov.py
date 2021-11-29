import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import inv, eigsh, expm

from qsolve.methods import (Method1D, Method2D)


class NUM1D(Method1D):

    def __init__(self, system, psi0, imtime=False, acoeff=20.0):

        super(NUM1D, self).__init__(system, psi0, imtime, acoeff)

        # Create the Numerov 4-th order Laplacian operator
        dt = system.t[1]-system.t[0]
        dx = system.x[1]-system.x[0]
        n = len(system.x)
        A = spdiags([[1.0]*n, [-2.0]*n, [1.0]*n], [-1, 0, 1], n, n,
                    format='csc')/dx**2
        B = spdiags([[1.0]*n, [10.0]*n, [1.0]*n], [-1, 0, 1], n, n,
                    format='csc')/12.0

        self.L = -inv(B)*A/(2*system.m)
        self.V = spdiags(system.V, [0], n, n, format='csc')
        self.H = self.L+self.V
        # Propagator
        self.prop = expm(-(1.0 if self.it else 1.0j)*self.H*dt)

    def step(self):

        self.psi *= self.prop
        self.psi *= self.border
        self.normalize()