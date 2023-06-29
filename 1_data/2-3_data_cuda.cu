#include <iostream>
#include <cmath>

__device__ double f(double x) {
    return pow(x, 2) - 3 * x + 2;
}

__global__ void calculateFunction(double start, double step, double *results) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double x = start + tid * step;
    results[tid] = f(x);
}

int main() {
    double start = 0.0;
    double end = 4.0;
    double step = 0.1;
    int numSteps = static_cast<int>((end - start) / step);

    // Allocate memory on the host
    double* hostResults = new double[numSteps];

    // Allocate memory on the device
    double* deviceResults;
    cudaMalloc((void**)&deviceResults, numSteps * sizeof(double));

    // Copy start and step values to the device
    cudaMemcpyToSymbol(start, &start, sizeof(double));
    cudaMemcpyToSymbol(step, &step, sizeof(double));

    // Launch kernel
    int blockSize = 256;
    int gridSize = (numSteps + blockSize - 1) / blockSize;
    calculateFunction<<<gridSize, blockSize>>>(start, step, deviceResults);

    // Copy results back from device to host
    cudaMemcpy(hostResults, deviceResults, numSteps * sizeof(double), cudaMemcpyDeviceToHost);

    // Print results
//    for (int i = 0; i < numSteps; ++i) {
//        double x = start + i * step;
//        std::cout << "f(" << x << ") = " << hostResults[i] << std::endl;
//    }

    // Clean up
    delete[] hostResults;
    cudaFree(deviceResults);

    return 0;
}

