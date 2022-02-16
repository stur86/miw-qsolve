import sys
import numpy as np

from qsolve import System1D, BPM1D, Runner, MIW2P, NUM1D

find_ground = False

k = 2.0
def V(x):
    return 0.5*k*x**2

sys1d = System1D(-5, 5, 100, 0.01, 1000, V)
psi0 = np.exp(-(sys1d.x-1.0)**2/4)
bpm = BPM1D(sys1d, psi0, imtime=find_ground)
miw = MIW2P(sys1d, psi0, imtime=find_ground)
nmv = NUM1D(sys1d, psi0, imtime=find_ground)

sfile = None
if len(sys.argv) > 1:
    sfile = sys.argv[1]

runner = Runner(sys1d, [bpm, miw, nmv])

runner.run(fps=120, plot_each=10, lim=(0,0.8), plot_V=True, 
           savefile=sfile)