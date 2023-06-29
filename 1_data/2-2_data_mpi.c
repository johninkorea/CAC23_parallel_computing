#include <iostream>
#include <cmath>
#include <mpi.h>

// Define your function here
// Modify this function according to your specific requirements
double f(double x) {
    return pow(x, 2) - 3 * x + 2;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start = 0.0;
    double end = 4.0;
    double step = 0.1;
    int numSteps = static_cast<int>((end - start) / (step * size));

    // Calculate local range for each process
    double localStart = start + rank * numSteps * step;
    double localEnd = localStart + numSteps * step;

    // Allocate memory for local results
    double* localResults = new double[numSteps];

    // Calculate local function values
    for (int i = 0; i < numSteps; ++i) {
        double x = localStart + i * step;
        localResults[i] = f(x);
    }

    // Gather results to the root process
    double* allResults = nullptr;
    if (rank == 0) {
        allResults = new double[numSteps * size];
    }
    MPI_Gather(localResults, numSteps, MPI_DOUBLE, allResults, numSteps, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Print results on the root process
    if (rank == 0) {
        for (int i = 0; i < numSteps * size; ++i) {
            double x = start + i * step;
            std::cout << "f(" << x << ") = " << allResults[i] << std::endl;
        }
        delete[] allResults;
    }

    // Clean up
    delete[] localResults;
    MPI_Finalize();

    return 0;
}

