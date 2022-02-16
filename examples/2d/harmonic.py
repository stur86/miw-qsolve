import sys
import numpy as np
from matplotlib import cm

from qsolve import System2D, BPM2D, MIW4P, MIW4P, Runner

find_ground = False

#K = np.diag([2.0, 3.0])
K11 = 4.0
K22 = 6.0
K12 = 2.0

K = np.array([[K11, K12], [K12, K22]])

print(np.linalg.eigh(K))

def V(xy):
    return 0.5*(xy@K@xy)

sys2d = System2D(-5, 5, 100, -5, 5, 100, 0.01, 1000, V)

psi0 = np.exp(-np.sum((sys2d.xy-1.0)**2, axis=2)/4)

bpm = BPM2D(sys2d, psi0, imtime=find_ground, plot_cmap=cm.hot)
miw = MIW4P(sys2d, psi0, imtime=find_ground, gamma=6.0)

sfile = None
if len(sys.argv) > 1:
    sfile = sys.argv[1]

runner = Runner(sys2d, [bpm, miw])

runner.run(fps=30, plot_each=10, plot_V=True, savefile=sfile)