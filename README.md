# architetture

This repository contains the project done for the course "Programming on parallel architectures" at the University of Udine.

It contains serial and parallel (CUDA) implementation of the DPLL algorithm for solving the problem:

-----------------------------------------------------------------------------------------------------------------------
Given a set of tuples (no-goods) find whether an assignment extists for the variables such that no no-good is falsified
-----------------------------------------------------------------------------------------------------------------------

It contains different folders with different CNF-models (different sizes) and their respective no-good-model counterparts. 

**Usage of no_good_solver.c: (compile, e.g. gcc no_good_solver.c -o no_good_solver), then call ./no_good_solver "\<path of the model\>"**

**Usage of no_good_solver_CUDA.CU: (compile, e.g.  nvcc no_good_solver_CUDA.cu -o no_good_solver_CUDA) then call ./no_good_solver_CUDA "\<path of the model\>"**

The python script can be modified in order to generate new test cases, both CNF-models and no no-good-models will be generated.

#WARNING: currently there are no checks on the existance of the file specified model, thus if you specify a non existing file or one without reading permission it's core dump.
