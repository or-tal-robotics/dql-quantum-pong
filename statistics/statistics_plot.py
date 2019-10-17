import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def smooth(x):
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i-99)
        y[i] = float(x[start:(i+1)].sum())/(i-start+1)
    return y

S40 = np.load("stat_quantum_choice_29092019(60).npy")
C40 = np.load("stat_quantum_choice_C_29092019(60).npy")

Sn = np.load("episode_rewards_30092019(70).npy") 

plt.suptitle('Quantum Pong (84X84X70)')

plt.subplot(2,2,1)
plt.plot(smooth(C40[:,0]), label='C00')
plt.plot(smooth(C40[:,1]), label='C01')
plt.plot(smooth(C40[:,2]), label='C10')
plt.plot(smooth(C40[:,3]), label='C11')
plt.xlabel("Episodes")
plt.ylabel("Frequency")
plt.legend()

plt.subplot(2,2,2)
plt.plot(smooth(C40[:,5]), label='P(Quantum state)')
plt.xlabel("Episodes")
plt.ylabel("P(Quantum state)")


plt.subplot(2,2,3)
plt.plot(S40[0,:1500], label='Agent 1 (quantum)')
plt.plot(S40[1,:1500], label='Agent 2 (quantum)')
plt.plot(smooth(Sn[0,:1500]), label='Agent 1 (no quantum)')
plt.plot(smooth(Sn[1,:1500]), label='Agent 2 (no quantum)')
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.legend()


plt.subplot(2,2,4)
board_type = 0.7
xy = np.array([[0,0],[0,0.84],[0.84,0.84],[0.84,0.84-board_type]])
currentAxis = plt.gca()
currentAxis.add_patch(Polygon(xy,alpha=1,edgecolor='black', facecolor='none'))
currentAxis.set_xlim(-0.1, 0.9)
currentAxis.set_ylim(-0.1, 0.9)
plt.axis('off')


plt.show()
