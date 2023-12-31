//#include <iostream>
#include <cmath>
//#include <time.h>
#include <chrono>
#include <cstdio>
// #include <sys/time.h>
//#include <ctime>

using namespace std;
using namespace chrono;
// using std::cout; using std::endl;

// Define your function here
// Modify this function according to your specific requirements
double f(double x) {
    return pow(x, 2) - 3 * x + 2;
}

int main() {
    system_clock::time_point start_time = system_clock::now();

    double sta = 0.0;
    double end = 10000.0;
    double step = 0.001;

    for (double x = sta; x < end; x += step) {
        double result = f(x);
        //cout << result << endl;
        //std::cout << "f(" << x << ") = " << result << std::endl;
    }
    system_clock::time_point end_time = system_clock::now();
    nanoseconds nano = end_time - start_time;
    printf("%d\n",nano.count() / 1000);
    return 0;
}

