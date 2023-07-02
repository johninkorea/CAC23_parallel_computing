# CAC23_parallel_computing

This repository documents the project we worked on the 14th KIAS CAC Summer School. We aimed to solve the [damped harmonic oscillator](https://beltoforion.de/en/harmonic_oscillator/) problem using [PINN](https://en.wikipedia.org/wiki/Physics-informed_neural_networks).


First, you can find the MPI code in the [MPI](https://github.com/johninkorea/CAC23_parallel_computing/tree/main/MPI) folder. The purpose of this code is to create an exact solution for plotting and training the network.


Second, the [PINN](https://github.com/johninkorea/CAC23_parallel_computing/tree/main/PINN) folder contains the code for Physics-Informed Neural Network (PINN) to solve the ODE equation.


Lastly, I tackled the same problem using the Bayesian Kernel, and you can find the code in the [Bayesian](https://github.com/johninkorea/CAC23_parallel_computing/tree/main/Bayesian) folder.
