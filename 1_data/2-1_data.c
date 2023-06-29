#include <iostream>
#include <cmath>
#include <time.h>
#include <chrono>
#include <sys/time.h>
#include <ctime>

using namespace std;
using namespace chrono;
using std::cout; using std::endl;

// Define your function here
// Modify this function according to your specific requirements
double f(double x) {
    return pow(x, 2) - 3 * x + 2;
}

int main() {
    system_clock::time_point start = system_clock::now();

    double sta = 0.0;
    double end = 4.0;
    double step = 0.1;

    for (double x = sta; x < end; x += step) {
        double result = f(x);
        //std::cout << "f(" << x << ") = " << result << std::endl;
    }
    system_clock::time_point end = system_clock::now();
    nanoseconds nano = end - start;
    cout << "경과시간(나노초 ns) : " << nano.count() << endl;
    return 0;
}

