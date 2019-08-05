# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

y1_no_leapfrog = np.load("y1_no_leapfrog.npy")
y2_no_leapfrog = np.load("y2_no_leapfrog.npy")
y1_leapfrog = np.load("y1_leapfrog.npy")
y2_leapfrog = np.load("y2_leapfrog.npy")

plt.plot(y1_no_leapfrog[:450], label='agent 1 - no leapfrog')
plt.plot(y2_no_leapfrog[:450], label='agent 2 - no leapfrog')
plt.plot(y1_leapfrog[:450], label='agent 1 - leapfrog')
plt.plot(y2_leapfrog[:450], label='agent 2 - leapfrog')
plt.legend()
plt.xlabel("Training episodes")
plt.ylabel("Reward")
plt.show()
