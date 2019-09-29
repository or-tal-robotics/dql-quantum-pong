import numpy as np
import matplotlib.pyplot as plt

def smooth(x):
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i-99)
        y[i] = float(x[start:(i+1)].sum())/(i-start+1)
    return y

S40 = np.load("stat_quantum_choice_27092019(40).npy")
C40 = np.load("stat_quantum_choice_C_27092019(40).npy")
S70 = np.load("stat_quantum_choice24092019(70).npy")
C70 = np.load("stat_quantum_choice_C24092019(70).npy")
Sn = np.load("stat_no_quantum_14092019.npy") 

plt.subplot(1,3,1)
plt.plot(smooth(C40[:,0]), label='C00')
plt.plot(smooth(C40[:,1]), label='C01')
plt.plot(smooth(C40[:,2]), label='C10')
plt.plot(smooth(C40[:,3]), label='C11')
plt.xlabel("Episodes")
plt.ylabel("Frequency")
plt.legend()

plt.subplot(1,3,2)
plt.plot(smooth(C40[:,5]), label='P(Quantum state)')
plt.xlabel("Episodes")
plt.ylabel("P(Quantum state)")


plt.subplot(1,3,3)
plt.plot(S40[0,:], label='Agent 1 (quantum 40)')
plt.plot(S40[1,:], label='Agent 2 (quantum 40)')
plt.plot(S70[0,:], label='Agent 1 (quantum 70)')
plt.plot(S70[1,:], label='Agent 2 (quantum 70)')
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.legend()

plt.show()
