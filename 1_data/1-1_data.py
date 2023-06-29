#import numpy as np
import datetime
start_time = datetime.datetime.now()

# def f(x):
#     #d, w0 = 1.5, 15
#     #w = np.sqrt(w0**2-d**2)
#     #phi = np.arctan(-d/w)
#     #A = 1/(2*np.cos(phi))
#     #cos = np.cos(phi+w*x)
#     #sin = np.sin(phi+w*x)
#     #exp = np.exp(-d*x)
#     #y  = exp*2*A*cos
#     y = np.pow(x, 2) - 3 * x + 2
#     return y

def f(x):
    return x**2 - 3*x + 2

# Calculate and print f(x) for 0 < x < 4
start = 0.0
end = 10000.0
step = 0.001

x = start
while x < end:
    result = f(x)
    #print("f({:.2f}) = {:.2f}".format(x, result))
    x += step
end_time = datetime.datetime.now()

elapsed_time = end_time - start_time
ms_elapsed_time = elapsed_time.microseconds

print(ms_elapsed_time)
