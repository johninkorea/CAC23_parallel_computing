from mpi4py import MPI

# Define your function here
# You can modify this function according to your specific requirements
def f(x):
    return x**2 - 3*x + 2

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Calculate local range for each process
start = 0.0
end = 4.0
step = 0.1
num_steps = int((end - start) / (step * size))

local_start = start + rank * num_steps * step
local_end = local_start + num_steps * step

# Calculate local function values
local_results = []
x = local_start
while x < local_end:
    result = f(x)
    local_results.append((x, result))
    x += step

# Gather results on the root process
all_results = comm.gather(local_results, root=0)
"""
# Print results on the root process
if rank == 0:
    for results in all_results:
        for x, result in results:
            print("f({:.2f}) = {:.2f}".format(x, result))
"""
# Finalize MPI
MPI.Finalize()

