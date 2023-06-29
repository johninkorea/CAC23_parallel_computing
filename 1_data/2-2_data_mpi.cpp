#include <iostream>
#include <cmath>
#include <mpi.h>
#include <chrono>

using namespace std;
using namespace chrono;
// Define your function here
// Modify this function according to your specific requirements
double f(double x) {
    return pow(x, 2) - 3 * x + 2;
}

int main(int argc, char* argv[]) {
    system_clock::time_point start_time = system_clock::now();

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start = 0.0;
    double end = 10000.0;
    double step = 0.001;
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

    delete[] localResults;
    MPI_Finalize();
    // Print results on the root process
    if (rank == 0) {
        /*
        for (int i = 0; i < numSteps * size; ++i) {
            double x = start + i * step;
            std::cout << "f(" << x << ") = " << allResults[i] << std::endl;
        }
        */
        system_clock::time_point end_time = system_clock::now();
        nanoseconds nano = end_time - start_time;
        int int_nanosec = nano.count();
        //int ms_unit = int(pow(10, 6));
        //int us_unit = int(pow(10, 3));
        //int ms_elapsed = int_nanosec / ms_unit;
        //int us_elapsed = (int_nanosec % ms_unit) /us_unit;
        //int ns_elapsed = (int_nanosec % ms_unit) %us_unit;

        // cout << "Elapsed Time : " << ms_elapsed << "ms " << us_elapsed << "us " << ns_elapsed << "ns" << endl;
        cout << int_nanosec << endl;
        delete[] allResults;
    }

    // Clean up

    return 0;
}

