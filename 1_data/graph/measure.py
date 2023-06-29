import os
import  numpy as np
import matplotlib.pyplot as plt


asd=os.listdir(".")

z=1
while z<len(asd):
    print(asd[z])
    time = np.loadtxt(f"./{asd[z]}", unpack=1).T
    plt.hist(time, bins=100)
    plt.show()
    z+=1





