import  numpy as np
import matplotlib.pyplot as plt
time = np.loadtxt("./log", unpack=1)

plt.hist(time, bins=1000)
plt.show()






