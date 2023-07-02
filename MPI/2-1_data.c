//#include <iostream>
//#include <cmath>
#include <time.h>
#include <stdio.h>
#include <math.h>
//#include <chrono>
//#include <cstdio>
// #include <sys/time.h>
//#include <ctime>

// using std::cout; using std::endl;

// Define your function here
// Modify this function according to your specific requirements
double f(double x) {
    return pow(x, 2) - 3 * x + 2;
}

int main() {
    clock_t start1, end1;
    start1 = clock();

    double sta = 0.0;
    double end = 10000.0;
    double step = 0.001;

    for (double x = sta; x < end; x += step) {
        double result = f(x);
        //cout << result << endl;
        //std::cout << "f(" << x << ") = " << result << std::endl;
    }
    end1 = clock();
    // nanoseconds nano = end1 - start1;
    printf("%f\n",((float)end1) / CLOCKS_PER_SEC * 1000000);
    return 0;
}

