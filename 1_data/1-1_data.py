#import numpy as np
import timeit

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



# Calculate and print f(x) for 0 < x < 4
if __name__=="__main__":
    def f(x):
        return x**2 - 3*x + 2
    start_time = timeit.default_timer() #datetime.datetime.now()
    #rnd = random.randint(0,1) / 1000
    start = 0.0
    end = 10000.0
    step = 0.001
    #start = 0.0
    #end = 10000.0
    #step = 0.001
    
    x = start
    while x < end:
        result = f(x)
        #print("f({:.2f}) = {:.2f}".format(x, result))
        #print(result)
        x += step
    end_time = timeit.default_timer()
    
    elapsed_time = round((end_time - start_time) * 10 ** 6, 3) # end_time - start_time
    #ms_elapsed_time = round((t_1 - t_0) * 10 ** 6, 3) # elapsed_time.microseconds
    
    #print(ms_elapsed_time)
    print(f"{elapsed_time}")


    
