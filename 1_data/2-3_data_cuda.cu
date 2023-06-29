#include <iostream>
#include <cmath>
#include <chrono>

using namespace std;
using namespace chrono;
using std::cout; using std::endl;

__device__ double f(double x) {
    return pow(x, 2) - 3 * x + 2;
}

__global__ void calculateFunction(double start, double step, double *results) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double x = start + tid * step;
    results[tid] = f(x);
}

int main() {
    system_clock::time_point start_time = system_clock::now();
    
    double start = 0.0;
    double end = 10000.0;
    double step = 0.001;
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
    // int blockSize = 256;
    int blockSize = 50000;
    // int gridSize = (numSteps + blockSize - 1) / blockSize;
    int gridSize = 200;
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
    
    system_clock::time_point end_time = system_clock::now();
    nanoseconds nano = end_time - start_time;
    cout << nano.count() / 1000 << endl;
    return 0;
}

