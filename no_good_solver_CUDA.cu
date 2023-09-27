//THE FOLLOWING PROGRAM in the current version can support only the use of one GPU
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include "common.h"
#include <cuda_profiler_api.h>
#include <cassert>
//apparently this is needed for cuda ON WINDOWS
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <driver_types.h>

//the following code (from https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api) is used for debugging
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,">>>>>>>>>>>>>>>>>>>>>>>>>GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//use: gpuErrchk( cudaPeekAtLastError() );


int readFile_allocateMatrix(const char*, struct NoGoodDataCUDA_host*, struct NoGoodDataCUDA_devDynamic*);
void printError(char[]);
void popualteMatrix(FILE*, struct NoGoodDataCUDA_host*, struct NoGoodDataCUDA_devDynamic*);
void allocateMatrix();
void getNumberOfThreadsAndBlocks(int *, int* ,int *,int *);
void deallocateHost(struct NoGoodDataCUDA_host*);
void deallocateCUDA(struct NoGoodDataCUDA_devDynamic*);

__global__ void pureLiteralCheck(int*, int*, int*, int*,int *);
__global__ void removeNoGoodSetsContaining(int*, int*, int*, int*, int*,int */*, int */);
__global__ void removeLiteralFromNoGoods(int*, int*,int*, int*, int*,int*);
__global__ void assingValue(int*,int *,int *);
__global__ void decreaseVarsAppearingInNGsatisfied(int*, int*, int* , int*);

//computes the sum of the elemnts of a vector, used to avoid the usage of AtomicAdd and to be able to use multiple blocks
__global__ void parallelSum(int*, int *);
__global__ void unitPropagation2(int*,int*, int*,int* ,int *,int *, int *,int*);
__global__ void chooseVar(int*, int*);

int getSMcores(struct cudaDeviceProp);
bool solve(struct NoGoodDataCUDA_devDynamic, struct NoGoodDataCUDA_host, int, int);
void storePrevStateOnDevice(struct NoGoodDataCUDA_devDynamic, int**, int**, int**, int**, int**, int**, int**);
void revert(struct NoGoodDataCUDA_devDynamic*, struct NoGoodDataCUDA_host*, int**, int**, int**, int**, int**, int**, int**);


//algorithm data:

int** matrix; //the matrix that holds the clauses
int* dev_matrix; //the matrix that holds the clauses on the device

int noVars = -1; //the number of vars in the model
__device__ int dev_noVars; //the number of vars in the model on the device

int noNoGoods = -1; //the no of clauses (initially) in the model
__device__ int dev_noNoGoods; //the no of clauses (initially) in the model on the device


__device__ int dev_varToAssign;
__device__ int dev_valueToAssing;
__device__ int chooseVarResult;

int* returningNGchanged;


//technical (GPU related, but on host) data:
struct cudaDeviceProp deviceProperties; //on WINDOWS it seems we need to add the "Struct"
int noOfVarsPerThread; //the number of variables that each thread will handle in unit propagation, so that each thread will deal with 32 byte of memory (1 mem. transfer)
__device__ int dev_noOfVarsPerThread;

int noNoGoodsperThread;
__device__ int dev_noNoGoodsperThread;

int threadsPerBlock; //the number of threads per block, we want to have the maximum no of warps  to utilize the full SM
__device__ int dev_threadsPerBlock;

int blocksToLaunch_VARS;
__device__ int dev_blocksToLaunch_VARS;

int blocksToLaunch_NG;
__device__ int dev_blocksToLaunch_NG;
//device auxiliary variables

int conflict = RESET_CONFLCIT;

int* SM_dev_conflict; //used to store whether the propagation caused a conflict (a vector of at most the number of SM in the device)
__device__ int dev_conflict;
//auxiliary small vectors to support the usage of more SMs
int* SM_dev_varsYetToBeAssigned;
int* SM_dev_currentNoGoods;

bool breakSearchAfterOne = true; //if true, the search will stop after the first solution is found
bool solutionFound = false; //if true, a solution was found, used to stop the search


int main(int argc, char const* argv[]) {

    cudaProfilerStart();

    //we just check, then GPUSno won't be used to scale the program on multiple devices
    int GPUSno;
    if (cudaGetDeviceCount(&GPUSno) != cudaSuccess) {
        //printError("No GPU detected");
        return -1;
    }

    //if the user didn't insert the file path or typed more
    if (argc != 2) {
        //printError("Insert the file path");
        return -2;
    }
   
    //create the strucure both on the device (without a struct) and on the host
    struct NoGoodDataCUDA_host data;
    struct NoGoodDataCUDA_devDynamic dev_data;
    
    //we get the properties of the GPU
    cudaGetDeviceProperties(&deviceProperties, 0);
    //we get the threads per block
    threadsPerBlock = getSMcores(deviceProperties);

    printf("The detected GPU has %d SMs each with %d cores\n", deviceProperties.multiProcessorCount, threadsPerBlock);
    printf("*************************************************\n");
    //we check if the GPU is supported, since we will launch at least 64 threads per block (in order to have 4 warps per block, and exploit the new GPUs)
    if (threadsPerBlock == 0) {
        printf("The GPU is not supported, buy a better one :)");
        return -3;
    }
    clock_t t;
    t = clock();

    //we populate it with the data from the file
    if(readFile_allocateMatrix(argv[1], &data, &dev_data)==-1){
        return 0;
    }


   
    //we populate the four varables
    getNumberOfThreadsAndBlocks(&blocksToLaunch_VARS,&blocksToLaunch_NG,&noOfVarsPerThread,&noNoGoodsperThread);

    cudaError_t err = cudaMemcpyToSymbol(dev_blocksToLaunch_NG, &blocksToLaunch_NG, sizeof(int), 0, cudaMemcpyHostToDevice);
    err = cudaMemcpyToSymbol(dev_blocksToLaunch_VARS, &blocksToLaunch_VARS, sizeof(int), 0, cudaMemcpyHostToDevice);
 

    printf("No of vars per th. %d, %d blocks will be launched \n", noOfVarsPerThread, blocksToLaunch_VARS);
    printf("No of no goods per th %d, %d blocks will be launched \n", noNoGoodsperThread, blocksToLaunch_NG);
    


    //**********************
    //USEFUL CODE STARTS HERE:
    //**********************

    //we copy the data to the device the two variables
    //thus we launch the number of blocks needed, each thread will handle noOfVarsPerThread variables (on newer GPUS 128 threads per block, four warps)

    //we launch the kernel that will handle the pure literals
    //here threads deal with vars
    err = cudaMemcpyAsync((dev_data.dev_lonelyVar), (data.lonelyVar), sizeof(int) * noNoGoods, cudaMemcpyHostToDevice);
    err = cudaMemcpyAsync((dev_data.dev_matrix_noGoodsStatus), (data.matrix_noGoodsStatus), sizeof(int) * noNoGoods, cudaMemcpyHostToDevice);
    err = cudaMemcpyAsync((dev_data.dev_noOfVarPerNoGood), (data.noOfVarPerNoGood), sizeof(int) * noNoGoods, cudaMemcpyHostToDevice);
    err = cudaMemcpyAsync((dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods), &(data.varsYetToBeAssigned), sizeof(int), cudaMemcpyHostToDevice);
    err = cudaMemcpyAsync((dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods + 1), &(data.currentNoGoods), sizeof(int), cudaMemcpyHostToDevice);
    err = cudaMemcpyAsync((dev_data.dev_varsAppearingInRemainingNoGoodsPositiveNegative), (data.varsAppearingInRemainingNoGoodsPositiveNegative), sizeof(int) * (noVars + 1)*2, cudaMemcpyHostToDevice);
    err = cudaMemcpyAsync((dev_data.dev_unitPropValuestoRemove), &(data.unitPropValuestoRemove), sizeof(int) * (noVars + 1), cudaMemcpyHostToDevice);

    gpuErrchk( cudaPeekAtLastError() );
    //pure literal check
    //***********************************
    pureLiteralCheck <<<blocksToLaunch_VARS, threadsPerBlock,threadsPerBlock*sizeof(int) >>> (dev_matrix, dev_data.dev_partialAssignment, dev_data.dev_varsAppearingInRemainingNoGoodsPositiveNegative, dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods, SM_dev_varsYetToBeAssigned);
    gpuErrchk( cudaPeekAtLastError() );
  
     //we launch just the threads we need, it may not fill a multiple of a warp
    parallelSum <<<1, blocksToLaunch_VARS, blocksToLaunch_VARS * sizeof(int) >>> (SM_dev_varsYetToBeAssigned,dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods);



    gpuErrchk( cudaPeekAtLastError() );
    //here threads deal with noGoods
    removeNoGoodSetsContaining <<<blocksToLaunch_NG, threadsPerBlock,threadsPerBlock*sizeof(int)>>> (dev_matrix,returningNGchanged, dev_data.dev_partialAssignment, dev_data.dev_matrix_noGoodsStatus, (dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods + 1), SM_dev_currentNoGoods/*, 1*/);
    gpuErrchk( cudaPeekAtLastError() );
    
    parallelSum <<<1, blocksToLaunch_NG, sizeof(int)* blocksToLaunch_NG >>> (SM_dev_currentNoGoods, (dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods+1));
   
    decreaseVarsAppearingInNGsatisfied<<<blocksToLaunch_VARS, threadsPerBlock>>>(dev_matrix,returningNGchanged,dev_data.dev_partialAssignment,dev_data.dev_varsAppearingInRemainingNoGoodsPositiveNegative);
    
    //************************************
    //end of pure literal check


    

    gpuErrchk( cudaPeekAtLastError() );
  





    //unit propagation
    //************************************
    //we call unit prop on the device, yes it's not "device work" still less consuming than copying back and forth though
    unitPropagation2<<<1,1>>>((dev_data.dev_unitPropValuestoRemove),dev_data.dev_matrix_noGoodsStatus,dev_data.dev_noOfVarPerNoGood,(dev_data.dev_partialAssignment),dev_data.dev_lonelyVar,dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods, dev_matrix,dev_data.dev_varsAppearingInRemainingNoGoodsPositiveNegative);

    removeNoGoodSetsContaining << <blocksToLaunch_NG, threadsPerBlock,threadsPerBlock*sizeof(int) >> > (dev_matrix, returningNGchanged, dev_data.dev_partialAssignment, dev_data.dev_matrix_noGoodsStatus, (dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods + 1), SM_dev_currentNoGoods /*, -1*/);
    
    parallelSum << <1, blocksToLaunch_NG, sizeof(int)* blocksToLaunch_NG >> > (SM_dev_currentNoGoods, (dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods + 1));


    decreaseVarsAppearingInNGsatisfied<<<blocksToLaunch_VARS, threadsPerBlock>>>(dev_matrix,returningNGchanged,dev_data.dev_partialAssignment,dev_data.dev_varsAppearingInRemainingNoGoodsPositiveNegative);

    //here threads deal with noGoods
    cudaDeviceSynchronize();
    removeLiteralFromNoGoods <<<blocksToLaunch_NG, threadsPerBlock,threadsPerBlock*sizeof(int)>>> (dev_matrix,dev_data.dev_matrix_noGoodsStatus, dev_data.dev_noOfVarPerNoGood, dev_data.dev_lonelyVar, dev_data.dev_partialAssignment,SM_dev_conflict);
    int* addr;
    cudaGetSymbolAddress((void**)&addr, dev_conflict); 
    gpuErrchk( cudaPeekAtLastError() );
    //cudaGetSymbolAddress((void**)&addrSM, SM_dev_conflict);   
    gpuErrchk( cudaPeekAtLastError() );
    err = cudaMemcpyFromSymbol(&(conflict), (dev_conflict), sizeof(int), 0, cudaMemcpyDeviceToHost);
    parallelSum <<<1 , blocksToLaunch_NG,sizeof(int)* blocksToLaunch_NG >>> (SM_dev_conflict, (addr));
    //************************************
    //end unit propagation
    

    
    //we copy just the few data we need on the host and we check
    err = cudaMemcpyAsync(&(data.varsYetToBeAssigned), (dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods), sizeof(int), cudaMemcpyDeviceToHost);
    err = cudaMemcpyAsync(&(data.currentNoGoods), (dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods + 1), sizeof(int), cudaMemcpyDeviceToHost);
    err = cudaMemcpyFromSymbol(&(conflict), (dev_conflict), sizeof(int), 0, cudaMemcpyDeviceToHost);


     //if we find a conlfict at the top level, the problem is unsatisfiable
   
    if (conflict != blocksToLaunch_NG*NO_CONFLICT) {
        printf("\n\n\n**********UNSATISFIABLE**********\n\n\n");
        //we free the cuda side
        deallocateCUDA(&dev_data);
        //the matrix (& vectors) on the host isn't needed anymore
        deallocateHost(&data);
        return 0;
    }

    //if we somehow already have an assignment, we can skip the search
    if (data.currentNoGoods == 0) {
        printf("\n\n\n**********SATISFIABLE**********\n\n\n");
        //we free the cuda side
        deallocateCUDA(&dev_data);
        //the matrix (& vectors) on the host isn't needed anymore
        deallocateHost(&data);

        return 0;
    }
    //************************************
    //end of unit propagation
    
    //again not a device work, but still faster than copy
    chooseVar<<<1,1>>>(dev_data.dev_partialAssignment, dev_data.dev_varsAppearingInRemainingNoGoodsPositiveNegative);
    
    int var;
    //copy back the choice
    cudaMemcpyFromSymbol(&(var), (chooseVarResult), sizeof(int), 0, cudaMemcpyDeviceToHost);
    
    

    if (solve(dev_data, data, var, TRUE)  || solve(dev_data, data, var, FALSE)) {
        printf("\n\n\n**********SATISFIABLE**********\n\n\n");
    }
    else {
        printf("\n\n\n**********UNSATISFIABLE**********\n\n\n");
    }
    cudaProfilerStop();
    //we free the cuda side
    deallocateCUDA(&dev_data);
    //the matrix (& vectors) on the host isn't needed anymore
    deallocateHost(&data);
    gpuErrchk( cudaPeekAtLastError() );
    //calculate and print the time elapsed
    t = clock() - t;
    double time_taken = ((double)t) / CLOCKS_PER_SEC; // in seconds
    printf("\n\n took %f seconds to execute \n", time_taken);


    return 0;
}
    
//reads the content of a simil DMACS file and populates the data structure
// (not the fanciest function but it's called just once)
int readFile_allocateMatrix(const char* str, struct NoGoodDataCUDA_host* data, struct NoGoodDataCUDA_devDynamic* dev_data) {

    FILE* ptr;
    char ch;
    ptr = fopen(str, "r");

      if (NULL == ptr) {
        printf("\nNo such file or can't be opened\n");
        return -1;
    }

    bool isComment = true;
    bool newLine = true;
    //we skip the comments
    while (isComment == true && !feof(ptr)) {
        ch = fgetc(ptr);

        //a comment
        if (ch == 'c' && newLine == true) {
            isComment = true;
        }
        if (ch == 'p' && newLine == true) {
            isComment = false;
        }

        if (ch == '\n') {
            newLine = true;
        }
        else {
            newLine = false;
        }
    }

    //skip over "p nogood"
    int i = 8;
    while (!feof(ptr) && i > 0) {
        ch = fgetc(ptr);
        i--;
    }

    //ignore return value for now
    fscanf(ptr, "%d", &noVars);
    fscanf(ptr, "%d", &noNoGoods);


    //we copy the no_goods and the no_vars on the device
    cudaError_t err = cudaMemcpyToSymbol(dev_noNoGoods, &noNoGoods, sizeof(int), 0, cudaMemcpyHostToDevice);
    err = cudaMemcpyToSymbol(dev_threadsPerBlock, &threadsPerBlock, sizeof(int), 0, cudaMemcpyHostToDevice);
    //printf("copy dev_threadsPerBlock%s\n",cudaGetErrorString(err) );
    err = cudaMemcpyToSymbol(dev_noVars, &noVars, sizeof(int), 0, cudaMemcpyHostToDevice);
    //printf("copy No vars%s\n",cudaGetErrorString(err) );

    data->currentNoGoods = noNoGoods;
    data->varsYetToBeAssigned = noVars;

    //we also copy the values in the struct on the device, as we did in the two lines above for the host
    err = cudaMalloc((void**)&(dev_data->dev_varsYetToBeAssigned_dev_currentNoGoods), 2 * sizeof(int));
    err = cudaMemcpyAsync((dev_data->dev_varsYetToBeAssigned_dev_currentNoGoods), &noVars, sizeof(int), cudaMemcpyHostToDevice);
    err = cudaMemcpyAsync((dev_data->dev_varsYetToBeAssigned_dev_currentNoGoods + 1), &noNoGoods, sizeof(int), cudaMemcpyHostToDevice);

    //WE Copy the value of the conflict on the device
    err = cudaMemcpyToSymbol(dev_conflict , &conflict, sizeof(int), 0, cudaMemcpyHostToDevice);
    popualteMatrix(ptr, data, dev_data);
    fclose(ptr);
    return 0;
}

//subprocedure called by readFile_allocateMatrix it populates the data structure and other arrays such as varBothNegatedAndNot
void popualteMatrix(FILE* ptr, struct NoGoodDataCUDA_host* data, struct NoGoodDataCUDA_devDynamic* dev_data) {

    //we allocate both matrices
    allocateMatrix();

    //we use pinned memory now, we allocate all the structures on the host in this way

    cudaHostAlloc((void**)&(data->noOfVarPerNoGood), noNoGoods * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&(data->lonelyVar), noNoGoods * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&(data->partialAssignment), (noVars + 1) * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&(data->varsAppearingInRemainingNoGoodsPositiveNegative), 2*(noVars + 1) * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&(data->unitPropValuestoRemove), (noVars + 1) * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&(data->matrix_noGoodsStatus), noNoGoods * sizeof(int), cudaHostAllocDefault);

    //we need to initialize the memory (shouldn't be needed, HOWEVER)
    for(int i=1; i<(noVars + 1); i++){
        data->unitPropValuestoRemove[i]=0;
        data->varsAppearingInRemainingNoGoodsPositiveNegative[i]=0;
        data->varsAppearingInRemainingNoGoodsPositiveNegative[i+(noVars)+1]=0;
        data->partialAssignment[i]=0;
    }
    //we initialize the rest of the data 
    for(int i=0; i<noNoGoods; i++){
        data->matrix_noGoodsStatus[i]=0;
        data->lonelyVar[i]=0;
        data->noOfVarPerNoGood[i]=0;
    }


    //we allocate device side

    cudaError_t err = cudaMalloc((void**)&(dev_data->dev_partialAssignment), (noVars + 1) * sizeof(int));
    err = cudaMalloc((void**)&(dev_data->dev_noOfVarPerNoGood), noNoGoods * sizeof(int));
    err = cudaMalloc((void**)&(dev_data->dev_matrix_noGoodsStatus), noNoGoods * sizeof(int));
    err = cudaMalloc((void**)&(dev_data->dev_lonelyVar), noNoGoods * sizeof(int));
    err = cudaMalloc((void**)&(dev_data->dev_varsAppearingInRemainingNoGoodsPositiveNegative), 2*(noVars + 1) * sizeof(int));
    err = cudaMalloc((void**)&(dev_data->dev_unitPropValuestoRemove), (noVars + 1) * sizeof(int));
    err = cudaMalloc((void**)&(returningNGchanged), noNoGoods * sizeof(int));

    /*for (int i = 0; i < noVars + 1; i++) {
        varBothNegatedAndNot[i] = FIRST_APPEARENCE;
    }*/

    int clauseCounter = 0;
    int literal = 0;
    while (!feof(ptr) && clauseCounter < noNoGoods) {

        //no idea why fscanf READS positive number as negative and vv (on Windows)
        fscanf(ptr, "%d", &literal);
        if (literal == 0) {
            data->matrix_noGoodsStatus[clauseCounter] = UNSATISFIED; //we set the clauses (as in the next line, to unsat)
            matrix[clauseCounter][0] = UNSATISFIED; //the first cell of the matrix is the status of the clause
            clauseCounter++;
        } else {

            int sign = literal > 0 ? POSITIVE_LIT : NEGATED_LIT;
            matrix[clauseCounter][literal * sign] = sign;
            data->noOfVarPerNoGood[clauseCounter]++;
            //if i have more vars i won't read this, so it can contain a wrong value (if the literal is just one the value will be correct)
            data->lonelyVar[clauseCounter] = literal * sign;
            //prima ho i negativi poi i positivi, per semplicità di accesso
            data->varsAppearingInRemainingNoGoodsPositiveNegative[(noVars + 1) * ((int)(1 + sign) / 2) + literal * sign]++;

            /*
            if (varBothNegatedAndNot[literal * sign] == FIRST_APPEARENCE)
                varBothNegatedAndNot[literal * sign] = sign;
            if (varBothNegatedAndNot[literal * sign] == APPEARS_ONLY_POS && sign == NEGATED_LIT)
                varBothNegatedAndNot[literal * sign] = APPEARS_BOTH;
            if (varBothNegatedAndNot[literal * sign] == APPEARS_ONLY_NEG && sign == POSITIVE_LIT)
                varBothNegatedAndNot[literal * sign] = APPEARS_BOTH;*/
        }
    }

    //we assign to true possible missing variables
    for (int i = 1; i < noVars + 1; i++) {
        data->partialAssignment[i] = UNASSIGNED;
        //if the var doesn't appear anywhere, nor negated nor positve we assign it to true (a caso)
        if (data->varsAppearingInRemainingNoGoodsPositiveNegative[i] + data->varsAppearingInRemainingNoGoodsPositiveNegative[i + (noVars + 1)] == 0) {
            data->partialAssignment[i] = TRUE;
            data->varsYetToBeAssigned--;
        }
    }

    //we now copy the content of the matrix to the device (https://forums.developer.nvidia.com/t/passing-dynamically-allocated-2d-array-to-device/43727 apparenlty works just for static matrices)
    for (int i = 0; i < noNoGoods; i++) {
        cudaError_t err = cudaMemcpy((dev_matrix + i * ((noVars + 1))), matrix[i], (noVars + 1) * sizeof(int), cudaMemcpyHostToDevice);
    }

    //we copy varBothNegatedAndNot
  
    //copy all the data
    cudaMemcpyAsync((dev_data->dev_partialAssignment), data->partialAssignment, sizeof(int) * (noVars + 1), cudaMemcpyHostToDevice);
    cudaMemcpyAsync((dev_data->dev_noOfVarPerNoGood), data->noOfVarPerNoGood, sizeof(int) * (noNoGoods), cudaMemcpyHostToDevice);
    cudaMemcpyAsync((dev_data->dev_matrix_noGoodsStatus), data->matrix_noGoodsStatus, sizeof(int) * (noNoGoods), cudaMemcpyHostToDevice);
    cudaMemcpyAsync((dev_data->dev_lonelyVar), data->lonelyVar, sizeof(int) * (noNoGoods), cudaMemcpyHostToDevice);
    cudaMemcpyAsync((dev_data->dev_varsAppearingInRemainingNoGoodsPositiveNegative), data->varsAppearingInRemainingNoGoodsPositiveNegative, sizeof(int) * (noVars + 1)*2, cudaMemcpyHostToDevice);

}
//allocates the matrix
void allocateMatrix() {
    //this won't be pinned, it's never transferred (except one time) by the GPU
    matrix = (int**)calloc(noNoGoods, sizeof(int*));
    //indeed arrays of pointers are not a good idea on the GPU
    cudaError_t err = cudaMalloc((void**)&dev_matrix, noNoGoods * (noVars + 1) * sizeof(int));

    for (int i = 0; i < noNoGoods; i++) {
        matrix[i] = (int*)calloc(noVars + 1, sizeof(int));
    }
}

//deallocates the matrix
void deallocateHost(struct NoGoodDataCUDA_host* data) {

    for (int i = 0; i < noNoGoods; i++) {
        free(matrix[i]);
    }
    free(matrix);
    //deallocate all the other stuff
    cudaFreeHost(data->partialAssignment);
    cudaFreeHost(data->noOfVarPerNoGood);
    cudaFreeHost(data->lonelyVar);
    cudaFreeHost(data->varsAppearingInRemainingNoGoodsPositiveNegative);
    cudaFreeHost(data->matrix_noGoodsStatus);
    cudaFreeHost(data->unitPropValuestoRemove);
   
}

//removes the literal (by assigning a value) from the no goods IF it's UNASSIGNED and shows up with only one sign (in the remaining no goods)
//one th per (constant no of) var
__global__ void  pureLiteralCheck(int* dev_matrix, int* dev_partialAssignment, int* dev_varsAppearingInRemainingNoGoodsPositiveNegative , int* dev_varsYetToBeAssigned, int* SM_dev_varsYetToBeAssigned) {
    
    //printf("here\n");
    int thPos = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ int decrease[];
    //block resets the counter
    __shared__ int valToDecrement;
    //the first thread of each block resets the counter for the block
    if (threadIdx.x == 0) {
        SM_dev_varsYetToBeAssigned[blockIdx.x]=0;
        valToDecrement = 0;

    }
    __syncthreads();
    register int i;
    decrease[threadIdx.x] = 0;

    //we scan each var (ths deal with vars)
    for (int c = 0; c < dev_noOfVarsPerThread; c++) {
        i = thPos + c * dev_threadsPerBlock*dev_blocksToLaunch_VARS;
        //we check varaible i
        //if (i <= dev_noVars && i!=0)
        //    printf("th: %d of block %d dealing with var: %d \n",thPos,blockIdx.x, i);
        
        if (i <= dev_noVars && i!=0 && (dev_partialAssignment)[i] == UNASSIGNED && (dev_varsAppearingInRemainingNoGoodsPositiveNegative[i] == 0 || dev_varsAppearingInRemainingNoGoodsPositiveNegative[i+(dev_noVars+1)] == 0)) {
            //printf("find in pureLiteralCheck\n");
            //we assign the proper value and then we decrease the counter for this thread in the block
            (dev_partialAssignment)[i] = (dev_varsAppearingInRemainingNoGoodsPositiveNegative[i]==0? FALSE:TRUE ); //can be done better
            decrease[threadIdx.x]--;
            //printf("ASSINGING A VAR in pl, var %d\n",i );
            //since we assigned a value we remove all such vars in the no goods
            dev_varsAppearingInRemainingNoGoodsPositiveNegative[i]=0;
            dev_varsAppearingInRemainingNoGoodsPositiveNegative[i+(dev_noVars+1)]=0; 
        }
        __syncthreads();
    }
    if (decrease[threadIdx.x] != 0) {
        //printf("we also decrease\n");
        atomicAdd(&(valToDecrement), decrease[threadIdx.x]);
    }

    __syncthreads();
    //the first thread of each block decreases the value
    if (threadIdx.x == 0) {
        atomicAdd((SM_dev_varsYetToBeAssigned+ blockIdx.x), valToDecrement);
        //printf("we decrement SM_dev_varsYetToBeAssigned[%d] to %d\n",blockIdx.x,valToDecrement);
    }

}

//removes (assigns 'falsified' satisfied) the no goods if they contain the literal varIndex with the indicated sign

//one th per no good
__global__ void removeNoGoodSetsContaining(int* matrix, int* returningNGchanged, int* dev_partialAssignment, int* dev_matrix_noGoodsStatus, int* dev_currentNoGoods,int * SM_dev_currentNoGoods/*, int flipSign*/) {

    int thPos = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ int decrease[];
    //block resets the counter
    __shared__ int valToDecrement;
    //the first thread of each block resets the counter for the block
    if(threadIdx.x==0){
        SM_dev_currentNoGoods[blockIdx.x]=0;
        valToDecrement=0;
    }
    __syncthreads();
    register int i;
    decrease[threadIdx.x] = 0;

    //we scan each no_good (each th scans reading the first cell of matrix and the relative var pos)
    for (int c = 0; c < dev_noNoGoodsperThread; c++) {
        i = thPos + c * dev_threadsPerBlock*dev_blocksToLaunch_NG;

        //first we reset the value (may be dirty since used in the prev. iteration) 
        returningNGchanged[i]=0;
        if (i < dev_noNoGoods) {
            //fixed a no good (thread) we loop on the row of the matrix (in this way ONLY ONE thead access each cell of the first column)
            for (int varIndex = 1; varIndex <= dev_noVars; varIndex++) {
                if (dev_partialAssignment[varIndex] != UNASSIGNED && *(matrix + i * (dev_noVars + 1) + varIndex) == -dev_partialAssignment[varIndex] && dev_matrix_noGoodsStatus[i] != SATISFIED) {
                    (dev_matrix_noGoodsStatus)[i] = SATISFIED;
                    decrease[threadIdx.x]--;
                    
                    //WE NEED TO DECREASE dev_varsAppearingInRemainingNoGoodsPositiveNegative according to the ones present in  [i] and we use the following return value, to call another kernel
                    returningNGchanged[i]=1;
                }
            }
        }
        __syncthreads();
    }
    if (decrease[threadIdx.x] != 0) {
        atomicAdd(&(valToDecrement), decrease[threadIdx.x]);
    }
    //non funzionerà con + blocchi
    __syncthreads();
    //the first thread of each block decreases the value
    if(threadIdx.x ==0){
        atomicAdd((SM_dev_currentNoGoods+ blockIdx.x), valToDecrement);
    }
}


//one th deals with one var (call after removeNoGoodSetsContaining)
__global__ void decreaseVarsAppearingInNGsatisfied(int* matrix, int* NGsthatChanged, int* partialAssignment, int* dev_varsAppearingInRemainingNoGoodsPositiveNegative) {
    


    int thPos = blockIdx.x * blockDim.x + threadIdx.x;

    for (int c = 0; c < dev_noOfVarsPerThread; c++) {
        int i = thPos + c * dev_threadsPerBlock*dev_blocksToLaunch_VARS;
        //we check varaible i
        if (i <= dev_noVars && i!=0){
            //for each NG we check if it has changed
            for(int ngIndex=0; ngIndex<dev_noNoGoods; ngIndex++){
                //if the NG changed and in that NG we had a literal
                if(NGsthatChanged[ngIndex]==1 && *(matrix + ngIndex * (dev_noVars + 1) + i)!=NO_LIT && partialAssignment[i]==UNASSIGNED){
                    //printf("literal sign: %d  (var: %d), 0 for negative, 1 for positve: %d\n",*(matrix + ngIndex * (dev_noVars + 1) + i),i ,((int)((1 + (*(matrix + ngIndex * (dev_noVars + 1) + i))) / 2)));
                    (dev_varsAppearingInRemainingNoGoodsPositiveNegative)[i + (dev_noVars + 1) * ((int)((1 + (*(matrix + ngIndex * (dev_noVars + 1) + i))) / 2))]--;
                }

            }
         
        }
        __syncthreads();
     }

}
//each th deals with a ng
__global__ void unitPropagation2(int* dev_unitPropValuestoRemove,int* dev_matrix_noGoodsStatus, int* dev_noOfVarPerNoGood,int* dev_partialAssignment,int *dev_lonelyVar,int *dev_varsYetToBeAssigned_dev_currentNoGoods, int *dev_matrix, int* dev_varsAppearingInRemainingNoGoodsPositiveNegative) {
    //printf("in kernel\n\n");

    for (int i = 1; i < dev_noVars + 1; i++) {
        (dev_unitPropValuestoRemove)[i] = 0; //we reset
    }
    //for each no good
    for (int i = 0; i < dev_noNoGoods; i++) {
        //if the no good is not satisfied and it has only one variable to assign we assign it
        if (dev_matrix_noGoodsStatus[i] == UNSATISFIED && dev_noOfVarPerNoGood[i] == 1 && dev_partialAssignment[dev_lonelyVar[i]] == UNASSIGNED) {

            //lonelyVar[i] is a column index
            //assing variable to value
            //printf("we assing var %d\n",dev_lonelyVar[i]);
            (dev_partialAssignment)[dev_lonelyVar[i]] = *(dev_matrix+i*(dev_noVars + 1)+ dev_lonelyVar[i]) > 0 ? FALSE : TRUE;
            (*dev_varsYetToBeAssigned_dev_currentNoGoods)--;
            (dev_varsAppearingInRemainingNoGoodsPositiveNegative)[dev_lonelyVar[i]]=0;
            (dev_varsAppearingInRemainingNoGoodsPositiveNegative)[dev_lonelyVar[i]+dev_noVars+1]=0;
          
        }
    }
}

//removes the literal from the nogood if the sign is the one indicated (part of unitPropagaition)
//Initially it has been kept on the host but after some modifications has been ported to the device side (it seems a bit forced but works, each thread scans at worst the number of vars squared)
__global__ void removeLiteralFromNoGoods(int* dev_matrix, int* dev_currentNoGoods,int* dev_noOfVarPerNoGood, int* dev_lonelyVar, int* partialAssignment,int* SM_dev_conflict) {
    //scan column (varIndex) of matrix

    extern __shared__ int currentLonely[];
    __shared__ int block_conflict;
    int thPos = blockIdx.x * blockDim.x + threadIdx.x;
    int i;
    //if we are the thread 0 we reset the globla conflict
    if(thPos==0){
        dev_conflict=RESET_CONFLCIT;
    }
    if(threadIdx.x==0){
        *(SM_dev_conflict+blockIdx.x)=NO_CONFLICT; //-3
        block_conflict=NO_CONFLICT;
        //printf("we initialize block %d to NO_CONFLICT, sono th %d del blocco %d \n",blockIdx.x,threadIdx.x,blockIdx.x);
    }
    __syncthreads();
    //each th deals with a ngdev_currentNoGoods
    for (int c = 0; c < dev_noNoGoodsperThread; c++) {
        i = thPos + c * dev_threadsPerBlock*dev_blocksToLaunch_NG;
        if (i < dev_noNoGoods) {
            //foreach var
            dev_noOfVarPerNoGood[i]=0;
            for (int varIndex = 1; varIndex <= dev_noVars; varIndex++) {
            
                if (block_conflict == CONFLICT) //this if doesn't cause divergence :) either all or none ths in the warp (better in the block) behave the same
                   return;
                
                if(partialAssignment[varIndex] == UNASSIGNED  && (*(dev_currentNoGoods + i)) == UNSATISFIED &&  (*(dev_matrix + i * (dev_noVars + 1) + varIndex))!= NO_LIT){
                    //printf("th %d increasing vars in noGood %d cause of var %d \n",thPos, i,varIndex );
                    dev_noOfVarPerNoGood[i]++;
                    //printf("thPOos < 191: %d\n",threadIdx.x);
                    currentLonely[threadIdx.x]=varIndex; //it will be used only if at the end dev_noOfVarPerNoGood[]=0 else, can contain whatever
                }
            }
            if(dev_noOfVarPerNoGood[i]==1 && (*(dev_currentNoGoods + i))== UNSATISFIED){
                dev_lonelyVar[i]=currentLonely[threadIdx.x];
                //printf("lonely upd\n");
            }
            if (dev_noOfVarPerNoGood[i] == 0 && (*(dev_currentNoGoods + i)) == UNSATISFIED) {
               
                atomicCAS(&block_conflict, NO_CONFLICT, CONFLICT);
                atomicCAS((SM_dev_conflict+blockIdx.x), NO_CONFLICT, CONFLICT);
                //printf("conflict lol cause ng: %d %d, block. %d %d\n",i,*(dev_matrix + i * (dev_noVars + 1)),block_conflict,*(SM_dev_conflict+blockIdx.x)); 
                //why do we use block_conflict if we also do atomicCAS on SM_dev_conflict? because the first if is faster in shared
            }
        }
        //__syncthreads();
    }
}


//returns the index of the first unassigned variable (more policies to be implemented)
__global__ void chooseVar(int* dev_partialAssignment, int* varsAppearingInRemainingNoGoodsPositiveNegative){
    //return the fist unassigned var

    for (int i = 1; i < dev_noVars + 1; i++) {
        if (dev_partialAssignment[i] == UNASSIGNED && (varsAppearingInRemainingNoGoodsPositiveNegative[i] + varsAppearingInRemainingNoGoodsPositiveNegative[i + dev_noVars + 1]) > 0) {
            chooseVarResult=i;
            return;
        }
    }
    //if all vars are assigned return -1 (never)
    printf("NEVER NERE :) \n");
    chooseVarResult=-1;
}

bool solve(struct NoGoodDataCUDA_devDynamic dev_data, struct NoGoodDataCUDA_host data, int var, int value) {
    
    //printf("currentLonelyrent no goods: %d, current vars yet: %d assign var: %d=%d\n", data.currentNoGoods, data.varsYetToBeAssigned,var,value );
    //gpuErrchk( cudaPeekAtLastError() );

    //if we want to stop after the first solution and it's already found
    if (solutionFound && breakSearchAfterOne){
        return true;
    }
   

    //gpuErrchk( cudaPeekAtLastError() );

    //local variables which will be used to revert the state of the data structure when backtracking
    int* dev_prevPartialAssignment = NULL;
    int* dev_prevNoOfVarPerNoGood = NULL;
    int* dev_prevLonelyVar = NULL;
    int* dev_prevVarsAppearingInRemainingNoGoods = NULL;
    int* dev_matrix_prevNoGoodsStatus = NULL; //the first column of the matrix is the status of the clause
    int* dev_prevVarsYetToBeAssigned_prevCurrentNoGoods = NULL;
    int* dev_prevUnitPropValuestoRemove = NULL;

    //allocates and copies the above arrays
    storePrevStateOnDevice(dev_data, &dev_prevPartialAssignment, &dev_prevNoOfVarPerNoGood, &dev_prevLonelyVar, &dev_matrix_prevNoGoodsStatus, &dev_prevVarsAppearingInRemainingNoGoods, &dev_prevVarsYetToBeAssigned_prevCurrentNoGoods, &dev_prevUnitPropValuestoRemove);
   
    //gpuErrchk( cudaPeekAtLastError() );
    cudaMemcpyToSymbol(dev_varToAssign, &var, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(dev_valueToAssing, &value, sizeof(int), 0, cudaMemcpyHostToDevice);
  
    //gpuErrchk( cudaPeekAtLastError() );

    //assigning and cleaning
    //***********************************
    //dev_varToAssign,dev_valueToAssing are gobal __device__
    assingValue<<<1,1>>>((dev_data.dev_partialAssignment),dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods,dev_data.dev_varsAppearingInRemainingNoGoodsPositiveNegative);

    removeNoGoodSetsContaining <<<blocksToLaunch_NG, threadsPerBlock,threadsPerBlock*sizeof(int)>>> (dev_matrix,returningNGchanged, dev_data.dev_partialAssignment, dev_data.dev_matrix_noGoodsStatus, (dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods + 1), SM_dev_currentNoGoods/*, 1*/);
    //gpuErrchk( cudaPeekAtLastError() );

    //removeNoGoodSetsContaining << <blocksToLaunch_NG, threadsPerBlock,threadsPerBlock*sizeof(int) >> > ( dev_data.dev_partialAssignment, dev_data.dev_matrix_noGoodsStatus, (dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods + 1), SM_dev_currentNoGoods, -1);
    
    parallelSum << <1, blocksToLaunch_NG, sizeof(int)* blocksToLaunch_NG >> > (SM_dev_currentNoGoods, (dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods + 1));
  
    decreaseVarsAppearingInNGsatisfied<<<blocksToLaunch_VARS, threadsPerBlock>>>(dev_matrix,returningNGchanged,dev_data.dev_partialAssignment,dev_data.dev_varsAppearingInRemainingNoGoodsPositiveNegative);
    //***********************************
    //end assigning and cleaning
    //gpuErrchk( cudaPeekAtLastError() );


    //pure literal check
    //***********************************
    pureLiteralCheck <<<blocksToLaunch_VARS, threadsPerBlock,threadsPerBlock*sizeof(int) >>> (dev_matrix, dev_data.dev_partialAssignment, dev_data.dev_varsAppearingInRemainingNoGoodsPositiveNegative, dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods, SM_dev_varsYetToBeAssigned);
     //we launch just the threads we need, it may not fill a multiple of a warp
    parallelSum <<<1, blocksToLaunch_VARS, blocksToLaunch_VARS * sizeof(int) >>> (SM_dev_varsYetToBeAssigned,dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods);

    //here threads deal with noGoods
    removeNoGoodSetsContaining <<<blocksToLaunch_NG, threadsPerBlock,threadsPerBlock*sizeof(int)>>> (dev_matrix,returningNGchanged, dev_data.dev_partialAssignment, dev_data.dev_matrix_noGoodsStatus, (dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods + 1), SM_dev_currentNoGoods/*, 1*/);
    parallelSum <<<1, blocksToLaunch_NG, sizeof(int)* blocksToLaunch_NG >>> (SM_dev_currentNoGoods, (dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods+1));
   
    decreaseVarsAppearingInNGsatisfied<<<blocksToLaunch_VARS, threadsPerBlock>>>(dev_matrix,returningNGchanged,dev_data.dev_partialAssignment,dev_data.dev_varsAppearingInRemainingNoGoodsPositiveNegative);
    //gpuErrchk( cudaPeekAtLastError() );

    //************************************
    //end of pure literal check



    //unit propagation
    //************************************
    //we call unit prop on the device, yes it's not "device work" still less consuming than copying back and forth though
    unitPropagation2<<<1,1>>>((dev_data.dev_unitPropValuestoRemove),dev_data.dev_matrix_noGoodsStatus,dev_data.dev_noOfVarPerNoGood,(dev_data.dev_partialAssignment),dev_data.dev_lonelyVar,dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods, dev_matrix,dev_data.dev_varsAppearingInRemainingNoGoodsPositiveNegative);

    removeNoGoodSetsContaining << <blocksToLaunch_NG, threadsPerBlock,threadsPerBlock*sizeof(int) >> > (dev_matrix, returningNGchanged, dev_data.dev_partialAssignment, dev_data.dev_matrix_noGoodsStatus, (dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods + 1), SM_dev_currentNoGoods /*, -1*/);
    parallelSum << <1, blocksToLaunch_NG, sizeof(int)* blocksToLaunch_NG >> > (SM_dev_currentNoGoods, (dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods + 1));


    decreaseVarsAppearingInNGsatisfied<<<blocksToLaunch_VARS, threadsPerBlock>>>(dev_matrix,returningNGchanged,dev_data.dev_partialAssignment,dev_data.dev_varsAppearingInRemainingNoGoodsPositiveNegative);


    removeLiteralFromNoGoods <<<blocksToLaunch_NG, threadsPerBlock,threadsPerBlock*sizeof(int)>>> (dev_matrix,dev_data.dev_matrix_noGoodsStatus, dev_data.dev_noOfVarPerNoGood, dev_data.dev_lonelyVar, dev_data.dev_partialAssignment,SM_dev_conflict);
    int* addr;
    cudaGetSymbolAddress((void**)&addr, dev_conflict);  

    parallelSum <<<1 , blocksToLaunch_NG,sizeof(int)* blocksToLaunch_NG >>> (SM_dev_conflict, (addr));


    cudaError_t err = cudaMemcpyAsync(&(data.varsYetToBeAssigned), (dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods), sizeof(int), cudaMemcpyDeviceToHost);
    err = cudaMemcpyAsync(&(data.currentNoGoods), (dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods + 1), sizeof(int), cudaMemcpyDeviceToHost);
    err = cudaMemcpyFromSymbol(&(conflict), (dev_conflict), sizeof(int), 0, cudaMemcpyDeviceToHost);
    //gpuErrchk( cudaPeekAtLastError() );

    //************************************
    //end unit propagation


    //if we find a conflict
    if (conflict != blocksToLaunch_NG*NO_CONFLICT) {
        revert(&dev_data, &data, &dev_prevPartialAssignment, &dev_prevNoOfVarPerNoGood, &dev_prevLonelyVar, &dev_matrix_prevNoGoodsStatus, &dev_prevVarsAppearingInRemainingNoGoods, &dev_prevVarsYetToBeAssigned_prevCurrentNoGoods, &dev_prevUnitPropValuestoRemove);
        return false;
    }

    //if the partialAssignment satisfies (falsifies) all the clauses we have found a solution
    if (data.currentNoGoods == 0) {
        printf("SATISFIABLE\n");
        //printf("Assignment:\n");
        //printVarArray(data.partialAssignment);
        solutionFound = true;
        return true;
    }


    // printf("solve:) current ng %d, current varsYetToBeAssigned %d\n", data.currentNoGoods,data.varsYetToBeAssigned);
      //if there are no more variables to assign (AND having previously checked that not all the no good are sat) we backtrack
    if (data.varsYetToBeAssigned == 0) {
        revert(&dev_data, &data, &dev_prevPartialAssignment, &dev_prevNoOfVarPerNoGood, &dev_prevLonelyVar, &dev_matrix_prevNoGoodsStatus, &dev_prevVarsAppearingInRemainingNoGoods, &dev_prevVarsYetToBeAssigned_prevCurrentNoGoods, &dev_prevUnitPropValuestoRemove);
        return false;
    }


    chooseVar<<<1,1>>>(dev_data.dev_partialAssignment, dev_data.dev_varsAppearingInRemainingNoGoodsPositiveNegative);
    int varTo;
    //copy back the choice
    cudaMemcpyFromSymbol(&(varTo), (chooseVarResult), sizeof(int), 0, cudaMemcpyDeviceToHost);
 

    //gpuErrchk( cudaPeekAtLastError() );

    //the check is done just for reverting purposes in case we need to backtrack
    if ((solve(dev_data, data, varTo, TRUE) || solve(dev_data, data, varTo, FALSE)) == false) {
        revert(&dev_data, &data, &dev_prevPartialAssignment, &dev_prevNoOfVarPerNoGood, &dev_prevLonelyVar, &dev_matrix_prevNoGoodsStatus, &dev_prevVarsAppearingInRemainingNoGoods, &dev_prevVarsYetToBeAssigned_prevCurrentNoGoods, &dev_prevUnitPropValuestoRemove);
        return false;
    }
    
    return false;
}



//CREDITS TO: https://stackoverflow.com/questions/32530604/how-can-i-get-number-of-cores-in-cuda-device
//returns the number of cores per SM, used to optimize the number of threads per block
int getSMcores(struct cudaDeviceProp devProp) {
    int cores = 0;
    switch (devProp.major) {
    case 3: // Kepler
        cores = 192;
        break;
    case 5: // Maxwell
        cores = 128;
        break;
    case 6: // Pascal
        if ((devProp.minor == 1) || (devProp.minor == 2)) cores = 128;
        else if (devProp.minor == 0) cores = 64;
        else printf("Unknown device type\n");
        break;
    case 7: // Volta and Turing
        if ((devProp.minor == 0) || (devProp.minor == 5)) cores = 64;
        else printf("Unknown device type\n");
        break;
    case 8: // Ampere
        if (devProp.minor == 0) cores = 64;
        else if (devProp.minor == 6) cores = 128;
        else if (devProp.minor == 9) cores = 128; // ada lovelace
        else printf("Unknown device type\n");
        break;
    case 9: // Hopper
        if (devProp.minor == 0) cores = 128;
        else printf("Unknown device type\n");
        break;
    default:
        printf("Unknown device type\n");
        break;
    }
    return cores;
}
//prints the content of the matrix (the first column is the status of each clause)
void deallocateCUDA(struct NoGoodDataCUDA_devDynamic *data_dev) {
    cudaFree(dev_matrix);
    
    cudaFree(data_dev->dev_partialAssignment);
    cudaFree(data_dev->dev_noOfVarPerNoGood);
    cudaFree(data_dev->dev_lonelyVar);
    cudaFree(data_dev->dev_varsAppearingInRemainingNoGoodsPositiveNegative);
    cudaFree(data_dev->dev_matrix_noGoodsStatus);
    cudaFree(data_dev->dev_unitPropValuestoRemove);
    cudaFree(data_dev->dev_varsYetToBeAssigned_dev_currentNoGoods);
    cudaFree(SM_dev_varsYetToBeAssigned);
    cudaFree(SM_dev_currentNoGoods);
    cudaFree(SM_dev_conflict);
}

void storePrevStateOnDevice(struct NoGoodDataCUDA_devDynamic dev_data, int** dev_prevPartialAssignment, int** dev_prevNoOfVarPerNoGood, int** dev_prevLonelyVar, int** dev_prevNoGoodStatus, int** dev_prevVarsAppearingInRemainingNoGoodsPositiveNegative, int** dev_prevVarsYetToBeAssigned_prevCurrentNoGoods, int** dev_prevUnitPropValuestoRemove) {
    //we allocate the space on the global memory for the following arrays:
    cudaError_t err = cudaMalloc((void**)dev_prevPartialAssignment, (noVars + 1) * sizeof(int));
    //printf("err alloc %s\n",cudaGetErrorString(err) );
    err = cudaMalloc((void**)dev_prevNoOfVarPerNoGood, noNoGoods * sizeof(int));
    //printf("err alloc %s\n",cudaGetErrorString(err) );
    err = cudaMalloc((void**)dev_prevLonelyVar, noNoGoods * sizeof(int));
    //printf("err alloc %s\n",cudaGetErrorString(err) );
    err = cudaMalloc((void**)dev_prevNoGoodStatus, noNoGoods * sizeof(int));
    //printf("err alloc %s\n",cudaGetErrorString(err) );
    err = cudaMalloc((void**)dev_prevVarsAppearingInRemainingNoGoodsPositiveNegative, (noVars + 1) * sizeof(int)*2);
    //printf("err alloc %s\n",cudaGetErrorString(err) );
    err = cudaMalloc((void**)dev_prevVarsYetToBeAssigned_prevCurrentNoGoods, 2 * sizeof(int));
    //printf("err alloc %s\n",cudaGetErrorString(err) );
    err = cudaMalloc((void**)dev_prevUnitPropValuestoRemove, (noVars + 1) * sizeof(int));
    //printf("err alloc %s\n",cudaGetErrorString(err) );
    //we copy the contents
    //according to: https://stackoverflow.com/questions/6063619/cuda-device-to-device-transfer-expensive to optimize the copy we could make another kernel
    err = cudaMemcpyAsync((*dev_prevPartialAssignment), dev_data.dev_partialAssignment, sizeof(int) * (noVars + 1), cudaMemcpyDeviceToDevice);
    //printf("err %s\n",cudaGetErrorString(err) );

    err = cudaMemcpyAsync((*dev_prevNoOfVarPerNoGood), dev_data.dev_noOfVarPerNoGood, sizeof(int) * (noNoGoods), cudaMemcpyDeviceToDevice);
    //printf("err %s\n",cudaGetErrorString(err) );
   
    err = cudaMemcpyAsync((*dev_prevLonelyVar), dev_data.dev_lonelyVar, sizeof(int) * (noNoGoods), cudaMemcpyDeviceToDevice);
    //printf("err %s\n",cudaGetErrorString(err) );
   
    err = cudaMemcpyAsync((*dev_prevNoGoodStatus), dev_data.dev_matrix_noGoodsStatus, sizeof(int) * noNoGoods, cudaMemcpyDeviceToDevice);
    //printf("err %s\n",cudaGetErrorString(err) );
   
    err = cudaMemcpyAsync((*dev_prevVarsAppearingInRemainingNoGoodsPositiveNegative), dev_data.dev_varsAppearingInRemainingNoGoodsPositiveNegative, sizeof(int) * (noVars + 1)*2, cudaMemcpyDeviceToDevice);
    //printf("err %s\n",cudaGetErrorString(err) );
   
    err = cudaMemcpyAsync((*dev_prevVarsYetToBeAssigned_prevCurrentNoGoods), (dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods), sizeof(int) * 2, cudaMemcpyDeviceToDevice);
    //printf("err %s\n",cudaGetErrorString(err) );
   
    err = cudaMemcpyAsync((*dev_prevUnitPropValuestoRemove), dev_data.dev_unitPropValuestoRemove, sizeof(int) * (noVars + 1), cudaMemcpyDeviceToDevice);
    //printf("err %s\n",cudaGetErrorString(err) );
   
}

//performs a copy of the arrays passed (to revert to the previous state) then it deallocates the memory
void revert(struct NoGoodDataCUDA_devDynamic* dev_data, struct NoGoodDataCUDA_host* data, int** dev_prevPartialAssignment, int** dev_prevNoOfVarPerNoGood, int** dev_prevLonelyVar, int** dev_noGoodStatus, int** dev_prevVarsAppearingInRemainingNoGoodsPositiveNegative, int** dev_prevVarsYetToBeAssigned_prevCurrentNoGoods, int** dev_prevUnitPropValuestoRemove) {
    cudaError_t err = cudaMemcpyAsync((dev_data->dev_partialAssignment), (*dev_prevPartialAssignment), sizeof(int) * (noVars + 1), cudaMemcpyDeviceToDevice);
    err = cudaMemcpyAsync((dev_data->dev_noOfVarPerNoGood), (*dev_prevNoOfVarPerNoGood), sizeof(int) * (noNoGoods), cudaMemcpyDeviceToDevice);
    err = cudaMemcpyAsync((dev_data->dev_lonelyVar), (*dev_prevLonelyVar), sizeof(int) * (noNoGoods), cudaMemcpyDeviceToDevice);
    err = cudaMemcpyAsync((dev_data->dev_matrix_noGoodsStatus), (*dev_noGoodStatus), sizeof(int) * noNoGoods, cudaMemcpyDeviceToDevice);
    err = cudaMemcpyAsync((dev_data->dev_varsAppearingInRemainingNoGoodsPositiveNegative), (*dev_prevVarsAppearingInRemainingNoGoodsPositiveNegative), sizeof(int) * (noVars + 1)*2, cudaMemcpyDeviceToDevice);
    err = cudaMemcpyAsync((dev_data->dev_varsYetToBeAssigned_dev_currentNoGoods), (*dev_prevVarsYetToBeAssigned_prevCurrentNoGoods), sizeof(int) * 2, cudaMemcpyDeviceToDevice);
    err = cudaMemcpyAsync((dev_data->dev_unitPropValuestoRemove), (*dev_prevUnitPropValuestoRemove), sizeof(int) * (noVars + 1), cudaMemcpyDeviceToDevice);
    cudaFree(*dev_prevPartialAssignment);
    cudaFree(*dev_prevNoOfVarPerNoGood);
    cudaFree(*dev_prevLonelyVar);
    cudaFree(*dev_prevVarsAppearingInRemainingNoGoodsPositiveNegative);
    cudaFree(*dev_noGoodStatus);
    cudaFree(*dev_prevVarsYetToBeAssigned_prevCurrentNoGoods);
    cudaFree(*dev_prevUnitPropValuestoRemove);
}

__global__ void assingValue(int *dev_partialAssignment,int *dev_varsYetToBeAssigned,int* dev_varsAppearingInRemainingNoGoodsPositiveNegative){
    (dev_partialAssignment)[dev_varToAssign]=dev_valueToAssing;
    (*dev_varsYetToBeAssigned)--;
    dev_varsAppearingInRemainingNoGoodsPositiveNegative[dev_varToAssign]=0;
    dev_varsAppearingInRemainingNoGoodsPositiveNegative[dev_varToAssign+(dev_noVars+1)]=0;

}

//a mini parallel reduction done in one step
__global__ void parallelSum(int* inArray, int* out) {
   extern __shared__ int s_array[];
    int thPos = blockIdx.x * blockDim.x + threadIdx.x;
    s_array[thPos] = inArray[threadIdx.x];
    __syncthreads();
    for (int i = (int) blockDim.x / 2; i > 0; i >>= 1) {
        if (thPos < i) {
            s_array[thPos] += s_array[thPos + i];
        }
        __syncthreads();
    }
    __syncthreads();
    if (thPos == 0) {
        //i don't change the whole method if inArray is odd, i deal with it here, we add the last (odd) element, if the block has > 1 th (thus if we use a odd number >1 than SMs)
        if(blockDim.x%2!=0 && blockDim.x > 1){
            *out = *out + s_array[blockDim.x-1];
        }
        *out=*out+ s_array[0];
    }
}

//returns the number of threads and blocks to launch for the different kernels, function it's just done to hide some code form the main procedure
void getNumberOfThreadsAndBlocks(int * blocksToLaunch_VARS, int* blocksToLaunch_NG,int *noOfVarsPerThread,int *noNoGoodsperThread){
     //THE FOLLOWING TWO FOR LOOPS ARE NOT OPTIMIZED AT ALL, but still they are really small loops executed once

    //we want at least one block per SM (in order to use the full capacity of the GPU)
    (*blocksToLaunch_VARS) = deviceProperties.multiProcessorCount;
    //if we have less variables than deviceProperties.multiProcessorCount*threadsPerBlock such that we can leave a SM empty and assigning just one var per thread we do so
    for (int i = 1; i <= deviceProperties.multiProcessorCount; i++) {
        if (i * threadsPerBlock > noVars) {
            (*blocksToLaunch_VARS) = i;
            break;
        }
    }

    //the same operation is done in order to get the blocks to launch for the kernels in which each thread deals with one or more no goods
    (*blocksToLaunch_NG) = deviceProperties.multiProcessorCount;
    for (int i = 1; i <= deviceProperties.multiProcessorCount; i++) {
        if (i * threadsPerBlock > noNoGoods) {
            (*blocksToLaunch_NG)  = i;
            break;
        }
    }


    //we get how many varibles and no goods each thread will handle
    (*noOfVarsPerThread) = ceil((float)noVars / ((threadsPerBlock) * deviceProperties.multiProcessorCount));
    (*noNoGoodsperThread) = ceil((float)noNoGoods / ((threadsPerBlock) * deviceProperties.multiProcessorCount));

    cudaMemcpyToSymbol(dev_noOfVarsPerThread, noOfVarsPerThread, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(dev_noNoGoodsperThread, noNoGoodsperThread, sizeof(int), 0, cudaMemcpyHostToDevice);

    //we now allocate the small arrays (one el for each block) that will be used to store the partial results of the kernels
    cudaError_t err=cudaMalloc((void**)&(SM_dev_varsYetToBeAssigned), (*blocksToLaunch_VARS) * sizeof(int));
    err=cudaMalloc((void**)&(SM_dev_currentNoGoods), (*blocksToLaunch_NG) * sizeof(int));

    err=cudaMalloc((void**)&(SM_dev_conflict), (*blocksToLaunch_NG)  * sizeof(int));

  
}
