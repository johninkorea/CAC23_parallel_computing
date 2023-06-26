#include <iostream>
#include <vector>

// CUDA kernel to add two vectors
__global__ void addVectorsCUDA(const int* vec1, const int* vec2, int* result, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        result[index] = vec1[index] + vec2[index];
    }
}

int main() {
    std::vector<int> vector1 = {1, 2, 3, 4, 5};
    std::vector<int> vector2 = {6, 7, 8, 9, 10};

    int size = vector1.size();
    std::vector<int> sum(size);

    // Allocate memory on the GPU
    int* deviceVec1;
    int* deviceVec2;
    int* deviceResult;
    cudaMalloc(&deviceVec1, size * sizeof(int));
    cudaMalloc(&deviceVec2, size * sizeof(int));
    cudaMalloc(&deviceResult, size * sizeof(int));

    // Copy input vectors from host to device
    cudaMemcpy(deviceVec1, vector1.data(), size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceVec2, vector2.data(), size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel on the GPU
    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    addVectorsCUDA<<<numBlocks, threadsPerBlock>>>(deviceVec1, deviceVec2, deviceResult, size);

    // Copy result from device to host
    cudaMemcpy(sum.data(), deviceResult, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(deviceVec1);
    cudaFree(deviceVec2);
    cudaFree(deviceResult);

    std::cout << "Result: ";
    for (const auto& value : sum) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    return 0;
}

