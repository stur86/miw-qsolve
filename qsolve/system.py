import numpy as np


class System(object):

    def __init__(self):
        pass


class System1D(System):

    def __init__(self, xmin, xmax, xsteps, dt, tsteps, V, m=1):

        self.x = np.linspace(xmin, xmax, xsteps)
        self.t = np.linspace(0, dt*tsteps, tsteps+1)
        self.V = V(self.x)
        self.m = m


class System2D(System):

    def __init__(self, xmin, xmax, xsteps, 
                       ymin, ymax, ysteps, 
                       dt, tsteps, V, m=1):

        self.x = np.linspace(xmin, xmax, xsteps)
        self.y = np.linspace(ymin, ymax, ysteps)
        self.xy = np.array(np.meshgrid(self.x, self.y, indexing='ij'))
        self.xy = np.moveaxis(self.xy, 0, 2)
        self.t = np.linspace(0, dt*tsteps, tsteps+1)
        self.V = np.apply_along_axis(V, 2, self.xy)
        self.m = m