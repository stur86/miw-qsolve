import numpy as np
from qsolve.system import System1D, System2D
from matplotlib import cm

class Method(object):

    def __init__(self, system, psi0, imtime=False, acoeff=20.0):

        self.system = system
        self.psi0 = np.array(psi0)+.0j
        self.psi = self.psi0.copy()
        self.it = imtime

        self.normalize()

        self.axes = None # For plotting
        self.graph = None

    def step(self):
        pass

    def norm(self):
        return np.linalg.norm(self.psi)

    def normalize(self):
        self.psi /= self.norm()

    @property
    def rho(self):
        return np.abs(self.psi)**2

    def plot(self):
        pass


class Method1D(Method):

    def __init__(self, system, psi0, imtime=False, acoeff=20.0):

        super(Method1D, self).__init__(system, psi0, imtime, acoeff)

        if not isinstance(system, System1D):
            raise ValueError('Method1D can only work with System1D')

        # Absorbing border
        x = system.x
        n = len(x)
        dx = x[1]-x[0]
        dt = system.t[1]-system.t[0]
        wx = dx*n/20
        xmax = x[-1]
        xmin = x[0]
        self.acoeff = acoeff
        if not self.it:
            self.border = np.exp(-acoeff*(2-np.tanh((x-xmin)/wx)+
                                 np.tanh((x-xmax)/wx))*dt)
        else:
            self.border = np.ones(n)

    def norm(self):
        return np.trapz(self.rho, self.system.x)**0.5        

    def plot(self):

        if self.axes:
            ax = self.axes
            if self.graph is None:
                self.graph = ax.plot(self.system.x, self.rho, 
                    label=self.__class__.__name__)[0]
            else:
                self.graph.set_ydata(self.rho)

        return [self.graph,]


class Method2D(Method):

    def __init__(self, system, psi0, imtime=False, acoeff=20.0, 
                 plot_cmap=cm.jet, plot_levels=5):

        super(Method2D, self).__init__(system, psi0, imtime, acoeff)

        if not isinstance(system, System2D):
            raise ValueError('Method2D can only work with System2D')

        # Absorbing border
        xy = system.xy
        nm = xy.shape[:2]
        dx = xy[1,0,0]-xy[0,0,0]
        dy = xy[0,1,1]-xy[0,0,1]
        dt = system.t[1]-system.t[0]
        wxy = np.array([dx, dy])*nm/20
        xymin = np.amin(xy, axis=(0,1))
        xymax = np.amax(xy, axis=(0,1))
        self.acoeff = acoeff
        if not self.it:
            arg1 = (xy-xymin[None,None])/wxy
            arg2 = (xy-xymax[None,None])/wxy
            self.border = np.exp(-acoeff*(2-np.tanh(arg1)+
                                 np.tanh(arg2))*dt)
            self.border = np.prod(self.border, axis=2)
        else:
            self.border = np.ones(nm)

        # Now for plotting
        self.cm = plot_cmap
        self.levels = plot_levels

    def norm(self):
        intx = np.trapz(self.rho, self.system.x, axis=0)
        norm = np.trapz(intx, self.system.y)**0.5
        return norm

    def plot(self):

        if self.axes:
            ax = self.axes
            if self.graph is not None:
                for c in self.graph.collections:
                    c.remove()
            self.graph = ax.contour(self.system.x, self.system.y, self.rho.T, 
                                    cmap=self.cm, levels=self.levels)

        return self.graph.collections