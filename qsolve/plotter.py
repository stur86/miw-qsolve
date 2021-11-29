import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class Plotter(object):

    def __init__(self, D=1):

        self.D = D # Dimension
        self.fig = None
        self.ax = None

    def start(self, lim=None):
        plt.ion()
        self.fig = plt.figure('QEvolve, {0}D'.format(self.D))
        self.ax = self.fig.add_subplot()

        if self.D == 2:
            self.ax.set_aspect('equal')

        if lim:
            if self.D == 1:
                self.ax.set_ylim(*lim)

    def plotV(self, system):

        if not self.fig:       
            raise RuntimeError('Figure not initialised')

        x = system.x
        V = system.V

        # Plot the potential
        if self.D == 1:
            lim = self.ax.get_ylim()
            V -= np.amin(V)
            V *= (lim[1]-lim[0])/np.amax(V)
            V += lim[0]
            self.ax.plot(x, V, label='V', lw=4, ls='dashed', 
                    c=(0,0,0,0.3))
        elif self.D == 2:
            y = system.y
            im = self.ax.pcolormesh(x, y, V, cmap=cm.gray_r, shading='nearest')

    def frame(self):
        if self.D == 1:
            self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()