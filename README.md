# Testing branch

This repository contains the code that can be used to test folder of models.

It contains both codes, for serial testing and parallel (CUDA) testing.

It contains different folders with different CNF-models (different sizes) and their respective no-good-model counterparts. 

**Usage of test_version.c: (compile, e.g. gcc test_version.c -o test_version), then call ./test_version**

**Usage of test_version_CUDA.cu: (compile, e.g.  nvcc test_version_CUDA.cu -o test_version_CUDA) then call ./test_version_CUDA**

## The test folders on which perform the tests must be set manually within the main functions of both files.

The python script can be modified in order to generate new test cases, both CNF-models and no no-good-models will be generated.
