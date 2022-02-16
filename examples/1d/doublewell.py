import sys
import numpy as np

from qsolve import System1D, BPM1D, Runner, MIW2P, MIW3P, NUM1D

find_ground = False

DE = -10.0
x0 = 2.0

def V(x):
    B = -2*DE/x0**2
    A = -B**2/(4*DE)
    return A*x**4-B*x**2

sys1d = System1D(-5, 5, 100, 0.003, 2000, V)
psi0 = np.exp(-(sys1d.x-x0)**2/2)
bpm = BPM1D(sys1d, psi0, imtime=find_ground)
miw2 = MIW2P(sys1d, psi0, imtime=find_ground)
miw3 = MIW3P(sys1d, psi0, imtime=find_ground)

sfile = None
if len(sys.argv) > 1:
    sfile = sys.argv[1]

runner = Runner(sys1d, [bpm, miw2, miw3])

runner.run(fps=120, plot_each=10, lim=(0,1.8), plot_V=True, savefile=sfile)