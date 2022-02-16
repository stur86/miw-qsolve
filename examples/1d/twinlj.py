import sys
import numpy as np

from qsolve import System1D, BPM1D, Runner, MIW2P, NUM1D, MIW3P

find_ground = False

DE = 10
r0 = 2.0
xhw = 5.0
def V(x):

    xl = x+xhw
    xr = xhw-x
    s = r0/2**(1/6)

    l6 = (s/xl)**6
    l12 = l6**2

    r6 = (s/xr)**6
    r12 = r6**2

    return 4*DE*(r12+l12-r6-l6)

sys1d = System1D(-xhw+1.5, xhw-1.5, 200, 0.003, 10000, V)
psi0 = np.exp(-(sys1d.x-(-xhw+r0))**2*10)
bpm = BPM1D(sys1d, psi0, imtime=find_ground)
mw2 = MIW2P(sys1d, psi0, imtime=find_ground)
mw3 = MIW3P(sys1d, psi0, imtime=find_ground)

sfile = None
if len(sys.argv) > 1:
    sfile = sys.argv[1]

runner = Runner(sys1d, [bpm, mw2, mw3])

runner.run(fps=120, plot_each=10, lim=(0,2.0), plot_V=True, savefile=sfile)