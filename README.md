# Testing branch

This repository contains the code that can be used to test folder of models.

It contains both codes, for serial testing and parallel (CUDA) testing.

It contains different folders with different CNF-models (different sizes) and their respective no-good-model counterparts. 

**Usage of test_serial.c: (compile, e.g. gcc test_serial.c -o test_serial), then call ./test_serial**

**Usage of test_advanced_startingPoint_CUDA.cu: (compile, e.g.  nvcc test_test_advanced_startingPoint_CUDA.cu -o test_version_CUDA) then call ./test_version_CUDA**

**The same goes for test_version_main.cu

## The test folders on which perform the tests must be set manually within the main functions of both files.

The python script can be modified in order to generate new test cases, both CNF-models and no no-good-models will be generated.
