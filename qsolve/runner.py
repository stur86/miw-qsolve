import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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

    def run(self, fps=60, plot_each=1, lim=(None, None), plot_V=False, 
            savefile=None):

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


        if savefile is None:
            # Render to screen

            for i, t in enumerate(self.system.t):
                t0 = time.time()
                pframe = (i%plot_each == 0)

                for m in self.methods:
                    m.step()
   
                if ax and pframe:
                    for m in self.methods:
                        m.plot()
                    plotter.frame()
                    t1 = time.time()
                    time.sleep(max(1.0/fps-t1+t0, 0))

        else: 

            def animate_step(frame_number):

                artists = []
                print(frame_number)

                for m in self.methods:
                    m.step()
                    artists += m.plot()

                return artists

            anim = animation.FuncAnimation(plotter.fig, animate_step, 
                            frames=len(self.system.t), blit=True)
            anim.save(savefile, fps=fps)
