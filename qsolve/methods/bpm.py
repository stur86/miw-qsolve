import numpy as np
from qsolve.methods import (Method1D, Method2D)


class BPM1D(Method1D):

    def __init__(self, system, psi0, imtime=False, acoeff=20.0):

        super(BPM1D, self).__init__(system, psi0, imtime, acoeff)

        # Now method specific stuff
        dt = system.t[1]-system.t[0]
        dx = system.x[1]-system.x[0]
        n = len(system.x)
        dt = -1.0j*dt if self.it else dt
        self.k = np.fft.fftfreq(n, dx)
        self.L = -(2*np.pi*self.k)**2/(2.0*system.m)
        self.linphase = np.exp(1.0j*self.L*dt)

        # Potential operator
        self.Vop = np.exp(-1.0j*system.V*dt)

    def step(self):

        self.psi *= self.Vop
        # Now the FFT step
        pfft = np.fft.fft(self.psi)
        pfft *= self.linphase
        self.psi = np.fft.ifft(pfft)*self.border

        # And renormalize
        self.normalize()

class BPM2D(Method2D):

    def __init__(self, system, psi0, imtime=False, acoeff=20.0, **kwargs):

        super(BPM2D, self).__init__(system, psi0, imtime, acoeff, **kwargs)

        dt = system.t[1]-system.t[0]
        dx = system.x[1]-system.x[0]
        dy = system.y[1]-system.y[0]
        nm = system.xy.shape
        dt = -1.0j*dt if self.it else dt
        self.kx = np.fft.fftfreq(nm[0], dx)
        self.ky = np.fft.fftfreq(nm[1], dy)
        self.k = np.array(np.meshgrid(self.kx, self.ky))
        self.k = np.moveaxis(self.k, 0, 2)
        self.L = -np.sum((2*np.pi*self.k)**2/(2.0*system.m), axis=2)
        self.linphase = np.exp(1.0j*self.L*dt)

        # Potential operator
        self.Vop = np.exp(-1.0j*system.V*dt)

    def step(self):

        self.psi *= self.Vop
        # Now the FFT step
        pfft = np.fft.fft2(self.psi)
        pfft *= self.linphase
        self.psi = np.fft.ifft2(pfft)*self.border

        # And renormalize
        self.normalize()