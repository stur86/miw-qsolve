import sys
import numpy as np

from qsolve import System1D, BPM1D, Runner, MIW2P, NUM1D, MIW3P

find_ground = True

DE = 15
r0 = 2.0
def V(x):

    s = r0/2**(1/6)
    p6 = (s/x)**6
    p12 = p6**2
    return 4*DE*(p12-p6)

sys1d = System1D(1.5, 10, 200, 0.01, 1000, V)
psi0 = np.exp(-(sys1d.x-r0)**2*10)
bpm = BPM1D(sys1d, psi0, imtime=find_ground)
mw2 = MIW2P(sys1d, psi0, imtime=find_ground)
mw3 = MIW3P(sys1d, psi0, imtime=find_ground)

runner = Runner(sys1d, [bpm, mw2, mw3])

sfile = None
if len(sys.argv) > 1:
    sfile = sys.argv[1]

runner.run(fps=120, plot_each=10, lim=(0,2.5), plot_V=True, savefile=sfile)