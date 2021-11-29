import numpy as np
from scipy.interpolate import UnivariateSpline, SmoothBivariateSpline

from qsolve.methods import (Method1D, Method2D)


class MIW2P(Method1D):

    def __init__(self, system, psi0, imtime=False, gamma=3.0):

        super(MIW2P, self).__init__(system, psi0, imtime)

        # Start by finding central point and sigma
        x = system.x
        rho0 = self.rho

        self.mu = np.trapz(rho0*x, x)
        self.sigma = np.trapz(rho0*(x-self.mu)**2, x)**0.5

        # Particles: positions and velocities
        self.px = np.array([self.mu-self.sigma, self.mu+self.sigma])
        self.pv = np.zeros(2)
        self.dt = system.t[1]-system.t[0]
        self.miwk = 1.0/(4*system.m)
        self.g = gamma

        # Interpolate potential
        self.V = UnivariateSpline(x, system.V, ext=3)
        self.dV = self.V.derivative()

        # For convenience when plotting
        self.ploth = np.amax(rho0)/3.0
        self.dbells = None

    def step(self):

        self.px += self.pv*self.dt/2.0

        self.mu = (self.px[1]+self.px[0])/2
        self.sigma = (self.px[1]-self.px[0])/2

        # Forces
        F = -self.dV(self.px)
        # Quantum force
        is3 = self.sigma**(-3)
        F += self.miwk*np.array([-is3, is3])
        # Damping if required
        if self.it:
            F -= self.g*self.pv

        self.pv += F*self.dt/self.system.m

        self.px += self.pv*self.dt/2.0

        self.mu = (self.px[1]+self.px[0])/2
        self.sigma = (self.px[1]-self.px[0])/2

        # And update the wavefunction...
        m, s = self.mu, self.sigma
        rho = np.exp(-(self.system.x-m)**2/(2*s**2))/((2*np.pi)**0.5*s)
        self.psi = rho**0.5

    def plot(self):

        super(self.__class__, self).plot()

        if self.axes:
            ax = self.axes
            if not self.dbells:
                self.dbells = ax.plot(self.px, [self.ploth, self.ploth], '-o')[0]
            else:
                self.dbells.set_xdata(self.px)

class MIW3P(Method1D):

    def __init__(self, system, psi0, imtime=False, gamma=3.0):

        super(MIW3P, self).__init__(system, psi0, imtime)

        # Start by finding central point and sigma
        x = system.x
        rho0 = self.rho

        self.mu = np.trapz(rho0*x, x)
        self.sigmas = np.trapz(rho0*(x-self.mu)**2, x)**0.5*np.ones(2)

        # Particles: positions and velocities
        self.px = np.array([self.mu-self.sigmas[0], self.mu, self.mu+self.sigmas[1]])
        self.pv = np.zeros(3)
        self.dt = system.t[1]-system.t[0]
        self.miwk = 1.0/(4*system.m)
        self.g = gamma

        # Interpolate potential
        self.V = UnivariateSpline(x, system.V, ext=3)
        self.dV = self.V.derivative()

        # For convenience when plotting
        self.ploth = np.amax(rho0)/3.0
        self.dbells = None

    def step(self):

        self.px += self.pv*self.dt/2.0

        self.mu = self.px[1]
        self.sigmas = np.diff(self.px)

        # Forces
        F = -self.dV(self.px)
        # Quantum force
        is3 = self.sigmas**(-3)
        F += self.miwk*np.array([-is3[0], is3[0]-is3[1], is3[1]])
        # Damping if required
        if self.it:
            F -= self.g*self.pv

        self.pv += F*self.dt/self.system.m

        self.px += self.pv*self.dt/2.0

        self.mu = self.px[1]
        self.sigmas = np.diff(self.px)

        # And update the wavefunction...
        m, (sl, sr) = self.mu, self.sigmas
        A = 1.0/((2*np.pi)**0.5*(sl+sr)/2)
        rho = A*np.where(self.system.x > m, np.exp(-(self.system.x-m)**2/(2*sr**2)), 
                         np.exp(-(self.system.x-m)**2/(2*sl**2)))
        self.psi = rho**0.5

    def plot(self):

        super(self.__class__, self).plot()

        if self.axes:
            ax = self.axes
            if not self.dbells:
                self.dbells = ax.plot(self.px, [self.ploth]*3, '-o')[0]
            else:
                self.dbells.set_xdata(self.px)

class MIW4P(Method2D):

    def __init__(self, system, psi0, imtime=False, gamma=3.0):

        super(MIW4P, self).__init__(system, psi0, imtime)

        # Start by finding central point and sigma
        x = system.x
        y = system.y
        xy = system.xy
        rho0 = self.rho

        rho0x = np.trapz(rho0, x, axis=0)
        rho0y = np.trapz(rho0, y, axis=1)

        mux = np.trapz(rho0y*x, x)
        muy = np.trapz(rho0x*y, y)

        self.mu = np.array([mux, muy])

        dxy =(xy-self.mu[None,None])
        dxy2 = dxy[:,:,None,:]*dxy[:,:,:,None]
        Smat = np.trapz(np.trapz(rho0[:,:,None,None]*dxy2, x, axis=0), y, axis=0)

        # Now find eigenvalues and eigenvectors of Smat
        evals, evecs = np.linalg.eigh(Smat)

        self.sigmas = evals**0.5
        self.evecs = evecs.T

        # Particles: positions and velocities
        self.px = np.array([self.mu + 2**0.5*d*s*a 
            for (s, a) in zip(self.sigmas, self.evecs) 
            for d in (-1, 1)])

        self.pv = np.zeros((4,2))
        self.dt = system.t[1]-system.t[0]
        self.miwk = 1.0/(2**1.5*system.m)
        self.g = gamma

        # Interpolate potential
        linxy = xy.reshape((-1,2))
        self.V = SmoothBivariateSpline(linxy[:,0], linxy[:,1], 
                                       system.V.reshape((-1,)))
        dVdx = lambda x, y: self.V(x, y, dx=1, grid=False)
        dVdy = lambda x, y: self.V(x, y, dy=1, grid=False)
        self.dV = lambda x, y: np.array([dVdx(x, y), dVdy(x, y)])

        self.dbells = None

    def step(self):

        self.px += self.pv*self.dt/2.0

        self.mu = np.average(self.px, axis=0)
        deltas = np.diff(self.px, axis=0)[[0,2]]
        norms = np.linalg.norm(deltas, axis=1)
        self.sigmas = norms/2**1.5
        self.evecs = deltas/norms[:,None]

        # Forces
        F = np.array([-self.dV(*p) for p in self.px])

        # Mean force
        mF = np.average(F, axis=0)
        # Stress forces
        stress = np.diff(F, axis=0)[[0,2]]
        stress = np.sum(stress*self.evecs, axis=1)[:,None]*self.evecs
        sF = np.array([-stress[0], stress[0], -stress[1], stress[1]])/2
        # Torque forces
        torque = np.average(np.cross(self.px-self.mu[None,:], F))

        s1, s2 = self.sigmas
        e1, e2 = self.evecs
        tF = np.array([-torque*e2/s1, torque*e2/s1, torque*e1/s2, -torque*e1/s2])/2**0.5

        # Quantum force
        is3 = self.sigmas**(-3)
        qF = self.miwk*np.array([-is3[0]*e1,
                                  is3[0]*e1, 
                                 -is3[1]*e2, 
                                  is3[1]*e2])
        F = mF + sF + tF + qF
        # Damping if required
        if self.it:
            F -= self.g*self.pv

        self.pv += F*self.dt/self.system.m
        self.px += self.pv*self.dt/2.0

        self.mu = np.average(self.px, axis=0)
        deltas = np.diff(self.px, axis=0)[[0,2]]
        norms = np.linalg.norm(deltas, axis=1)
        self.sigmas = norms/2**1.5
        self.evecs = deltas/norms[:,None]
        # Now make sure it's all orthogonal
        e1, e2 = self.evecs
        e2 -= (e1@e2)*e1
        self.evecs[1] = e2/np.linalg.norm(e2)

        e1, e2 = self.evecs
        s1, s2 = self.sigmas

        self.px = np.array([-s1*e1, s1*e1, -s2*e2, s2*e2])*2**0.5+self.mu[None,:]

        # And update the wavefunction...
        (s1, s2), (e1, e2) = self.sigmas, self.evecs
        xy = self.system.xy-self.mu[None,None]
        iS = e1[:,None]*e1[None,:]/s1**2 + e2[:,None]*e2[None,:]/s2**2
        kernel = np.sum((xy@iS)*xy, axis=2)
        rho = np.exp(-0.5*kernel)/(2*np.pi*np.linalg.norm(self.sigmas))
        self.psi = rho**0.5

    def plot(self):

        super(self.__class__, self).plot()

        if self.axes:
            ax = self.axes
            if not self.dbells:
                db1 = ax.plot(self.px[:2,0], self.px[:2,1], '-o', c=(0,0,0))[0]
                db2 = ax.plot(self.px[2:,0], self.px[2:,1], '-o', c=(0,0,0))[0]
                self.dbells = [db1, db2]
            else:
                db1, db2 = self.dbells
                db1.set_xdata(self.px[:2,0])
                db1.set_ydata(self.px[:2,1])
                db2.set_xdata(self.px[2:,0])            
                db2.set_ydata(self.px[2:,1])