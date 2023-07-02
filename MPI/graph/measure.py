import os
import  numpy as np
import matplotlib.pyplot as plt

os.system('rm asd.png')

asd=os.listdir(".")

mad=[]
times=[]
lab=[]

z=1
while z<len(asd):
    print(asd[z])
    time = np.loadtxt(f"./{asd[z]}", unpack=1).T
    
    times.append(time)
    mad.append(np.median(time))
    lab.append(asd[z][:-8])

    plt.hist(time, bins=1, label=asd[z][:-7])
    z+=1


# bins=np.linspace(0,3.4e6, 20)
# plt.hist(times, bins, label=[f'CUDA(1e5,1e2): {round(mad[0],2)}', f'python: {round(mad[1],2)}', f'python mpi(3): {round(mad[2],2)}', f'C++: {round(mad[3],2)}', f'C++ mpi(2): {round(mad[4],2)}'])
# plt.xscale('log')
plt.xlabel(r'Time [$\mu$s]')# [log scale]')
plt.ylabel('Count')
# plt.legend()
plt.savefig("asd", dpi=400)
plt.show()
print(mad)




