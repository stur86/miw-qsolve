import time
import numpy as np
import matplotlib.pyplot as plt

from qsolve.system import (System1D, System2D)
from qsolve.methods import (Method, Method1D, Method2D)
from qsolve.plotter import Plotter

class Runner(object):

    def __init__(self, system, methods=[]):

        self.system = system

        if isinstance(methods, Method):
            methods = [methods]

        self.methods = methods

        # Identify dimensionality
        self.D = {System1D: 1, System2D: 2}[system.__class__]

    def run(self, fps=60, plot_each=1, lim=(None, None), plot_V=False):

        if fps > 0:
            plotter = Plotter(self.D)
            plotter.start(lim=lim)

            if plot_V:
                plotter.plotV(self.system)

            ax = plotter.ax
        else:
            ax = None

        for m in self.methods:
            m.axes = ax

        for i, t in enumerate(self.system.t):
            t0 = time.time()
            pframe = (i%plot_each == 0)
            for m in self.methods:
                m.step()

                if ax and pframe:
                    m.plot()

            if ax and pframe:
                plotter.frame()
                t1 = time.time()
                time.sleep(max(1.0/fps-t1+t0, 0))