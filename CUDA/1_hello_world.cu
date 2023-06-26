#include <iostream>

// CUDA kernel function to print "Hello, World!" on each thread
__global__ void helloWorld() {
    printf("Hello, World! from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}

int main() {
    // Launch the kernel with a single block and 10 threads
    helloWorld<<<1, 10>>>();

    // Wait for GPU to finish execution
    cudaDeviceSynchronize();

    // Check for any errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    return 0;
}

