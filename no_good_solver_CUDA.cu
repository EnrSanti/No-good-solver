//THE FOLLOWING PROGRAM in the current version can support only the use of one GPU
#include <stdio.h>
#include <string.h>
#include <stdbool.h> 
#include <stdlib.h>
#include <math.h>
#include "common.h"

//apparently this is needed for cuda ON WINDOWS
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <driver_types.h>


void readFile_allocateMatrix(const char*, struct NoGoodDataCUDA_host*, struct NoGoodDataCUDA_devDynamic*);
void printError(char[]);
void popualteMatrix(FILE*, struct NoGoodDataCUDA_host*,struct NoGoodDataCUDA_devDynamic*);
void allocateMatrix();
void deallocateMatrix();
void freeCUDA();
__global__ void pureLiteralCheck(int*, int *, int *);
__global__ void removeNoGoodSetsContaining(int*, int*,int *,int *);
__global__ void unitPropagation(int* ,int * ,int * , int * ,int* ,int *);
__global__ void removeLiteralFromNoGoods(int*, int*,int*,int *,int* );
void printMatrix(int**);
int getSMcores(struct cudaDeviceProp);
bool solve(struct NoGoodDataCUDA_host,int,int);
int chooseVar(int *, int*);
void storePrevStateOnDevice();

//algorithm data:

//a piece of the former "NoGoodData" (the other piece, "the dynamic" one is declared locally), it contains two integer varaiables statically allocated on the device
__device__ int dev_varsYetToBeAssigned; //the number of variables that are not yet assigned
__device__ int dev_currentNoGoods; //the number of non satisfied clauses (yet)

int** matrix; //the matrix that holds the clauses
int* dev_matrix; //the matrix that holds the clauses on the device

int noVars = -1; //the number of vars in the model
__device__ int dev_noVars; //the number of vars in the model on the device

int noNoGoods = -1; //the no of clauses (initially) in the model
__device__ int dev_noNoGoods; //the no of clauses (initially) in the model on the device

int* varBothNegatedAndNot = NULL; //a int array that holds the status of the variables in the clauses (see the defines above)
int* dev_varBothNegatedAndNot=NULL;

//technical (GPU related, but on host) data:
struct cudaDeviceProp deviceProperties; //on WINDOWS it seems we need to add the "Struct" 
int noOfVarsPerThread; //the number of variables that each thread will handle in unit propagation, so that each thread will deal with 32 byte of memory (1 mem. transfer)
int noNoGoodsperThread;
int threadsPerBlock; //the number of threads per block, we want to have the maximum no of warps  to utilize the full SM
int conflict=NO_CONFLICT;

//device auxiliary variables
__device__ int dev_noOfVarsPerThread;
__device__ int dev_noNoGoodsperThread;
__device__ int dev_conflict; //used to store whether the propagation caused a conflict
int* dev_unitPropValuestoRemove; //used to store on the device the signs (found by unit propagation) of the variables to eliminate, in the serial version this wasn't necessary since removeNoGoodSetsContaining was called for each variable inside of a loop in unitProp


bool breakSearchAfterOne = false; //if true, the search will stop after the first solution is found
bool solutionFound = false; //if true, a solution was found, used to stop the search


int main(int argc, char const* argv[]) {

    //create the strucure both on the device (without a struct) and on the host
    struct NoGoodDataCUDA_host data;
    //__device__ struct NoGoodDataCUDA_devStatic dev_data_static; this has to be global, __device__ can't be declared local
    struct NoGoodDataCUDA_devDynamic dev_data_dynamic;

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

    //we get the properties of the GPU
    cudaGetDeviceProperties(&deviceProperties,0);
    //we get the threads per block
    threadsPerBlock = getSMcores(deviceProperties);
    printf("The detected GPU has %d SMs each with %d cores\n", deviceProperties.multiProcessorCount,threadsPerBlock);
    printf("*************************************************\n");
    //we check if the GPU is supported, since we will launch at least 64 threads per block (in order to have 4 warps per block, and exploit the new GPUs)
    if (threadsPerBlock==0) {
        //printError("The GPU is not supported, buy a better one :)");
		return -3;
    }

    //we populate it with the data from the file
    readFile_allocateMatrix(argv[1], &data,&dev_data_dynamic );
    //the matrix on the host isn't needed anymore
    deallocateMatrix();
    //THE FOLLOWING TWO FOR LOOPS ARE NOT OPTIMIZED AT ALL, but still they are really small loops executed once

    //we want at least one block per SM (in order to use the full capacity of the GPU)
    int blocksToLaunch_VARS = deviceProperties.multiProcessorCount;
    //if we have less variables than deviceProperties.multiProcessorCount*threadsPerBlock such that we can leave a SM empty and assigning just one var per thread we do so
    for(int i=1; i<=deviceProperties.multiProcessorCount; i++){
        if(i*threadsPerBlock>noVars){
            blocksToLaunch_VARS=i;
            break;
        }
    }

    //the same operation is done in order to get the blocks to launch for the kernels in which each thread deals with one or more no goods
    int blocksToLaunch_NG= deviceProperties.multiProcessorCount;
    for(int i=1; i<=deviceProperties.multiProcessorCount; i++){
        if(i*threadsPerBlock>noNoGoods){
            blocksToLaunch_NG=i;
            break;
        }
    }

    //we get how many varibles and no goods each thread will handle 
    noOfVarsPerThread = ceil((float)noVars / (threadsPerBlock * deviceProperties.multiProcessorCount));
    noNoGoodsperThread = ceil((float)noNoGoods / (threadsPerBlock * deviceProperties.multiProcessorCount));

    printf("No of vars per th. %d, %d blocks will be launched \n", noOfVarsPerThread,blocksToLaunch_VARS);
    printf("No of no goods per th %d, %d blocks will be launched \n", noNoGoodsperThread,blocksToLaunch_NG);
    //we copy the data to the device the two variables
    cudaMemcpyToSymbol(dev_noOfVarsPerThread, &noOfVarsPerThread, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(dev_noNoGoodsperThread, &noNoGoodsperThread, sizeof(int), 0, cudaMemcpyHostToDevice);

    //thus we launch the number of blocks needed, each thread will handle noOfVarsPerThread variables (on newer GPUS 128 threads per block, four warps)

    //**********************
    //USEFUL CODE STARTS HERE:
    //**********************

    //we launch the kernel that will handle the pure literals
    //here threads deal with vars
    pureLiteralCheck<<<blocksToLaunch_VARS, threadsPerBlock>>>(dev_matrix,dev_data_dynamic.dev_partialAssignment, dev_varBothNegatedAndNot);
 	cudaDeviceSynchronize();
    //here threads deal with noGoods
    removeNoGoodSetsContaining<<<blocksToLaunch_NG, threadsPerBlock>>>(dev_matrix, dev_varBothNegatedAndNot,dev_varBothNegatedAndNot,dev_data_dynamic.dev_matrix_noGoodsStatus);
    cudaDeviceSynchronize();
    //we do the unit propagation, and we subsequently remove the no goods and literals 
    unitPropagation<<<blocksToLaunch_NG, threadsPerBlock>>>(dev_matrix,dev_data_dynamic.dev_partialAssignment, dev_varBothNegatedAndNot,dev_data_dynamic.dev_noOfVarPerNoGood, dev_data_dynamic.dev_lonelyVar,dev_unitPropValuestoRemove);
    cudaDeviceSynchronize();
    removeNoGoodSetsContaining<<<blocksToLaunch_NG, threadsPerBlock>>>(dev_matrix, dev_varBothNegatedAndNot,dev_unitPropValuestoRemove, dev_data_dynamic.dev_matrix_noGoodsStatus);
    cudaDeviceSynchronize();
    //here threads deal with noGoods
    removeLiteralFromNoGoods<<<blocksToLaunch_NG, threadsPerBlock>>>(dev_matrix,dev_data_dynamic.dev_noOfVarPerNoGood, dev_data_dynamic.dev_lonelyVar,dev_data_dynamic.dev_partialAssignment,dev_unitPropValuestoRemove);


    //to check whether a conflict occurred we get the data:
    cudaError_t err=cudaMemcpyFromSymbol(&conflict,dev_conflict, sizeof(int));
    //printf("copy conflict%s\n",cudaGetErrorString(err) );
    err=cudaMemcpyFromSymbol(&(data.currentNoGoods),dev_currentNoGoods, sizeof(int));
    err=cudaMemcpyFromSymbol(&(data.varsYetToBeAssigned),dev_varsYetToBeAssigned, sizeof(int));

    cudaMemcpy(data.matrix_noGoodsStatus, dev_data_dynamic.dev_matrix_noGoodsStatus, (noNoGoods) * sizeof(int), cudaMemcpyDeviceToHost);
    for(int i=0; i<noNoGoods; i++){
        if (data.matrix_noGoodsStatus[i] == UNSATISFIED)
            printf("UNSATISFIED \n");
        else
            printf("SATISFIED   \n");
    } 

    if (conflict==CONFLICT) {
        printf("\n\n\n**********UNSATISFIABLE**********\n\n\n");
        freeCUDA();
        return 0;
    }

    if(data.currentNoGoods == 0) {
        printf("\n\n\n**********SATISFIABLE**********\n\n\n");
        printf("Assignment:\n");
        freeCUDA();
        return 0;
    }


    cudaMemcpy(data.partialAssignment, dev_data_dynamic.dev_partialAssignment, (noVars+1) * sizeof(int), cudaMemcpyDeviceToHost);
    
    //we choose a variable and we start the search
    int varToAssign = chooseVar(data.partialAssignment,data.varsAppearingInRemainingNoGoods);

    if (solve(data, varToAssign, TRUE) || solve(data, varToAssign, FALSE)) {
        printf("\n\n\n**********SATISFIABLE**********\n\n\n");
    }else {
        printf("\n\n\n**********UNSATISFIABLE**********\n\n\n");
    }

   
    freeCUDA();

    return 0;
}

//reads the content of a simil DMACS file and populates the data structure
// (not the fanciest function but it's called just once)
void readFile_allocateMatrix(const char* str, struct NoGoodDataCUDA_host* data,struct NoGoodDataCUDA_devDynamic* dev_data_dynamic) {

    FILE* ptr;
    char ch;
    ptr = fopen(str, "r");

    if (NULL == ptr) {
        //printError("No such file or can't be opened");
        return;
    }

    bool isComment = true;
    bool newLine = true;
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
    printf("\nnumber of vars: %d \n", noVars);
    printf("number of nogoods: %d \n", noNoGoods);
    int noNoGoodsperThread = ceil((float) (noNoGoods / (64 * 2)));

    cudaError_t err=cudaMemcpyToSymbol(dev_noNoGoods, &noNoGoods, sizeof(int), 0, cudaMemcpyHostToDevice);
    //printf("copy No goods%s\n",cudaGetErrorString(err) );
    err=cudaMemcpyToSymbol(dev_noVars, &noVars, sizeof(int), 0, cudaMemcpyHostToDevice);
	//printf("copy No vars%s\n",cudaGetErrorString(err) );

    data->currentNoGoods = noNoGoods;
    data->varsYetToBeAssigned = noVars;   

    err=cudaMemcpyToSymbol((dev_currentNoGoods), &noNoGoods, sizeof(int), 0, cudaMemcpyHostToDevice);
    err=cudaMemcpyToSymbol((dev_varsYetToBeAssigned), &noVars, sizeof(int), 0, cudaMemcpyHostToDevice);
    err=cudaMemcpyToSymbol(dev_conflict, &conflict, sizeof(int), 0, cudaMemcpyHostToDevice);
    popualteMatrix(ptr, data,dev_data_dynamic);
    fclose(ptr);
}

//subprocedure called by readFile_allocateMatrix it populates the data structure and other arrays such as varBothNegatedAndNot
void popualteMatrix(FILE* ptr, struct NoGoodDataCUDA_host* data,struct NoGoodDataCUDA_devDynamic* dev_data_dynamic) {

    allocateMatrix();

    varBothNegatedAndNot = (int*)calloc(noVars + 1, sizeof(int));
    cudaError_t err=cudaMalloc((void **)&dev_varBothNegatedAndNot, (noVars + 1) * sizeof(int));
	//printf("allocated varBothNegatedAndNot %s\n",cudaGetErrorString(err) );
    data->noOfVarPerNoGood = (int*)calloc(noNoGoods, sizeof(int));
    data->lonelyVar = (int*)calloc(noNoGoods, sizeof(int));
    data->partialAssignment = (int*)calloc(noVars + 1, sizeof(int));
    data->varsAppearingInRemainingNoGoods=(int *)calloc(noVars + 1, sizeof(int));
    
    data->matrix_noGoodsStatus= (int*)calloc(noNoGoods, sizeof(int));
    
    err=cudaMalloc((void**)&(dev_data_dynamic->dev_partialAssignment), (noVars + 1) * sizeof(int));
    //printf("allocated dev_partialAssignment %s\n",cudaGetErrorString(err) );
    err=cudaMalloc((void**)&(dev_data_dynamic->dev_noOfVarPerNoGood), noNoGoods* sizeof(int));
	//printf("allocated dev_noOfVarPerNoGood %s\n",cudaGetErrorString(err) );
    err=cudaMalloc((void**)&(dev_data_dynamic->dev_matrix_noGoodsStatus), noNoGoods * sizeof(int));
    err=cudaMalloc((void**)&(dev_data_dynamic->dev_lonelyVar), noNoGoods * sizeof(int));
    err=cudaMalloc((void**)&(dev_data_dynamic->dev_varsAppearingInRemainingNoGoods), (noVars + 1) * sizeof(int));
    err=cudaMalloc((void**)&dev_unitPropValuestoRemove, (noVars + 1) * sizeof(int));
    
    for (int i = 0; i < noVars + 1; i++) {
        varBothNegatedAndNot[i] = FIRST_APPEARENCE;
    }

    int clauseCounter = 0;
    int literal = 0;
    while (!feof(ptr) && clauseCounter < noNoGoods) {

        //no idea why fscanf READS positive number as negative and vv (on Windows) 
        fscanf(ptr, "%d", &literal);
        if (literal == 0) {
            matrix[clauseCounter][0] = UNSATISFIED; //the first cell of the matrix is the status of the clause
            clauseCounter++;
        } else {

            int sign = literal > 0 ? POSITIVE_LIT : NEGATED_LIT;
            matrix[clauseCounter][literal * sign] = sign;
            data->noOfVarPerNoGood[clauseCounter]++;
            //if i have more vars i won't read this, so it can contain a wrong value (if the literal is just one the value will be correct)
            data->lonelyVar[clauseCounter] = literal * sign;
            data->varsAppearingInRemainingNoGoods[literal*sign]++;

            //populate the varBothNegatedAndNot array
            if (varBothNegatedAndNot[literal * sign] == FIRST_APPEARENCE)
                varBothNegatedAndNot[literal * sign] = sign;
            if (varBothNegatedAndNot[literal * sign] == APPEARS_ONLY_POS && sign == NEGATED_LIT)
                varBothNegatedAndNot[literal * sign] = APPEARS_BOTH;
            if (varBothNegatedAndNot[literal * sign] == APPEARS_ONLY_NEG && sign == POSITIVE_LIT)
                varBothNegatedAndNot[literal * sign] = APPEARS_BOTH;
        }
    }

    //we assign to true possible missing variables
    for(int i=1; i<noVars+1; i++){
        if(data->varsAppearingInRemainingNoGoods[i]==0){
            data->partialAssignment[i]=TRUE;
            data->varsYetToBeAssigned--;
        }
    }
    //we now copy the content of the matrix to the device (https://forums.developer.nvidia.com/t/passing-dynamically-allocated-2d-array-to-device/43727 apparenlty works just for static matrices)
 	for(int i = 0; i < noNoGoods; i++) {
		cudaError_t err= cudaMemcpy((dev_matrix+i* ((noVars + 1) )), matrix[i], (noVars + 1) * sizeof(int), cudaMemcpyHostToDevice);
        data->matrix_noGoodsStatus[i] = UNSATISFIED;
    }

    //we copy varBothNegatedAndNot
    cudaMemcpy(dev_varBothNegatedAndNot , varBothNegatedAndNot, sizeof(int)* (noVars + 1), cudaMemcpyHostToDevice);
  	
    //printf("copied dev_varBothNegatedAndNot %s\n",cudaGetErrorString(err) );
    cudaMemcpy((dev_data_dynamic->dev_partialAssignment), data->partialAssignment, sizeof(int) * (noVars + 1), cudaMemcpyHostToDevice);    
    cudaMemcpy((dev_data_dynamic->dev_noOfVarPerNoGood), data->noOfVarPerNoGood, sizeof(int) * (noNoGoods), cudaMemcpyHostToDevice);
    cudaMemcpy((dev_data_dynamic->dev_matrix_noGoodsStatus), data->matrix_noGoodsStatus, sizeof(int) * (noNoGoods), cudaMemcpyHostToDevice);
    cudaMemcpy((dev_data_dynamic->dev_lonelyVar), data->lonelyVar, sizeof(int) * (noNoGoods), cudaMemcpyHostToDevice);
}

//prints str with "ERROR" in front of it
void printError(char str[]) {
    printf("ERROR: %s \n", str);
}

//allocates the matrix
void allocateMatrix() {
    matrix = (int**)calloc(noNoGoods, sizeof(int*));
    //indeed arrays of pointers are not a good idea on the GPU
    cudaError_t err=cudaMalloc((void **) &dev_matrix, noNoGoods * (noVars + 1) * sizeof(int));

    //printf("allocated matrix %s\n",cudaGetErrorString(err) );
    for (int i = 0; i < noNoGoods; i++) {
        matrix[i] = (int*)calloc(noVars + 1, sizeof(int));
    }

}

//deallocates the matrix
void deallocateMatrix() {

    for (int i = 0; i < noNoGoods; i++) {
        free(matrix[i]);
    }
    free(matrix);
    
}

//removes the literal (by assigning a value) from the no goods IF it's UNASSIGNED and shows up with only one sign (in the remaining no goods)
//one th per (constant no of) var
__global__ void  pureLiteralCheck(int* dev_matrix,int * dev_partialAssignment,int * dev_varBothNegatedAndNot) {
 
    int thPos = blockIdx.x * blockDim.x + threadIdx.x;
    int i;
    int decrease=0;
    //printf("Th. %d, no gooods: %d\n",thPos,dev_currentNoGoods);
    //we scan each var (ths deal with vars)
    for (int c = 0; c <dev_noOfVarsPerThread; c++) {
        i=thPos+c*dev_noOfVarsPerThread;
    	if (i<=dev_noVars && dev_partialAssignment[i] == UNASSIGNED && (dev_varBothNegatedAndNot[i] == APPEARS_ONLY_POS || dev_varBothNegatedAndNot[i] == APPEARS_ONLY_NEG)) {
            printf("Th. %d operating on var: %d \n", thPos,i);
	    	//printf("th. no %d working on pos %d \n",thPos,i);
	        dev_partialAssignment[i] = -dev_varBothNegatedAndNot[i];
	        //printf("th. no %d assigning to var %d\n",thPos,dev_varBothNegatedAndNot[i]);
	        //TODO substiture with one decrement at the end (e.g. warp level reduction)
	        decrease--;
	        
	        //this can't be called here, it would need too much serialization
	        //removeNoGoodSetsContaining(i,&(dev_matrix), &(dev_currentNoGoods), dev_varBothNegatedAndNot[i]);
    	}
    	__syncthreads();
    }   
    if(decrease!=0)
        atomicAdd(&(dev_varsYetToBeAssigned),decrease);
}

//removes (assigns 'falsified' satisfied) the no goods if they contain the literal varIndex with the indicated sign

//one th per no good
__global__ void removeNoGoodSetsContaining(int* matrix, int* dev_varBothNegatedAndNot,int* sing,int* dev_matrix_noGoodsStatus) {
	
	int thPos = blockIdx.x * blockDim.x + threadIdx.x;
  
	//printf("th %d dev_noNoGoods %d \n",thPos,dev_noNoGoods);         
    //we scan each no_good (each th scans reading the first cell of matrix and the relative var pos)
    
    int decrease=0;
    int i;
    for (int c = 0; c < dev_noNoGoodsperThread; c++) {
	    i=thPos+c*dev_noNoGoodsperThread;

        if (i<dev_noNoGoods){
	    	//fixed a no good (thread) we loop on the row of the matrix (in this way ONLY ONE thead access each cell of the first column)
	    	for(int varIndex=1; varIndex<=dev_noVars; varIndex++){

		    	if(sing[varIndex] !=0 && *(matrix+i*(dev_noVars+1)+varIndex) == sing[varIndex] && *(matrix + i*(dev_noVars+1)) !=SATISFIED) {
			        //remove the nogood set
			        printf("Th. %d, accessing no good:%d, SAT value: %d\n",thPos,i,*(matrix + i*(dev_noVars+1)));
                    *(matrix + i*(dev_noVars+1)) = SATISFIED; //è atomic 
			        dev_matrix_noGoodsStatus[i]=SATISFIED;
			        //TODO substiture with one decrement at the end (e.g. warp level reduction)
			       	decrease--;

			    }
			    __syncthreads();
		    }
		}
    }
    if(decrease!=0)
        atomicAdd(&(dev_currentNoGoods), decrease);
}


__global__ void unitPropagation(int* dev_matrix,int * dev_partialAssignment,int * dev_varBothNegatedAndNot,int *dev_noOfVarPerNoGood,int* dev_lonelyVar, int* dev_unitPropValuestoRemove) {
    
    int thPos = blockIdx.x * blockDim.x + threadIdx.x;
    int decrease=0;
    int i;
    //for each no good (th deals with ng)
    for (int c = 0; c < dev_noNoGoodsperThread; c++) {
        i=thPos+c*dev_noNoGoodsperThread;
        //printf("UNIT PROP: th %d at no NOOD: %d\n",thPos,i);
    	//printf("th %d lookin at cell %d \n",thPos,i );
    	if(i<dev_noNoGoods && *(dev_matrix+i*(dev_noVars+1)) == UNSATISFIED && dev_noOfVarPerNoGood[i] == 1){
            //printf("UNIT PROP: at no good: %d, seing: %d (-2 = UNSATISFIED)\n",i,*(dev_matrix+i*(dev_noVars+1)));
            //lonelyVar[i] is a column index

            dev_partialAssignment[dev_lonelyVar[i]] = *(dev_matrix+i*(dev_noVars+1)+dev_lonelyVar[i]) > 0 ? FALSE : TRUE;
            decrease--;
            dev_unitPropValuestoRemove[dev_lonelyVar[i]]=dev_partialAssignment[dev_lonelyVar[i]] == TRUE ? NEGATED_LIT : POSITIVE_LIT;

        }
        __syncthreads();

    }
    if(decrease!=0)
        atomicAdd(&(dev_varsYetToBeAssigned), decrease);

}
//removes the literal from the nogood if the sign is the one indicated (part of unitPropagaition)
//Initially it has been kept on the host but after some modifications has been ported to the device side (it seems a bit forced but works, each thread scans at worst the number of vars squared) 
__global__ void removeLiteralFromNoGoods(int* dev_matrix, int* dev_noOfVarPerNoGood,int* dev_lonelyVar,int* dev_partialAssignment, int *dev_sign) {
    //scan column (varIndex) of matrix
    int thPos = blockIdx.x * blockDim.x + threadIdx.x;
    int i;
    for (int c = 0; c < dev_noNoGoodsperThread; c++) {
        i=thPos+c*dev_noNoGoodsperThread;
        if (i<dev_noNoGoods){
            for(int varIndex=1; varIndex<=dev_noVars; varIndex++){
                if(dev_conflict!=NO_CONFLICT)
                    return;
                if (dev_sign[varIndex] !=0 && *(dev_matrix+i*(dev_noVars+1)+varIndex) == -dev_sign[varIndex]) {
                    //remove the literal
                    dev_noOfVarPerNoGood[i]--;
                    if(dev_noOfVarPerNoGood[i]==1){
                        //search and assing the literal to the lonelyVar
                        for (int j = 1; j < dev_noVars + 1; j++) {
                            if (*(dev_matrix+i*(dev_noVars+1)+j) != NO_LIT && dev_partialAssignment[j]==UNASSIGNED) {
                                dev_lonelyVar[i] = j;
                            }
                        }
                    }
                    if (dev_noOfVarPerNoGood[i] == 0) {
                        atomicAdd(&dev_conflict,1);
                        //dev_conflict=CONFLICT;
                    }
                }
                __syncthreads();
            }
            __syncthreads();
        }
    }  
}


//returns the index of the first unassigned variable (more policies to be implemented)
int chooseVar(int *partialAssignment, int* varsAppearingInRemainingNoGoods) {
    //return the fist unassigned var
    for (int i = 1; i < noVars + 1; i++) {
        if (partialAssignment[i] == UNASSIGNED && varsAppearingInRemainingNoGoods[i]>0) {
            return i;
        }
    }
    //if all vars are assigned return -1 (never)
    return -1;
}

bool solve(struct NoGoodDataCUDA_host data, int var, int value) {
    
    //if we want to stop after the first solution and it's already found
    if (solutionFound && breakSearchAfterOne)
        return true;
    
    //local variables which will be used to revert the state of the data structure when backtracking
    //int* dev_prevPartialAssignment = NULL;
    //int* dev_prevNoOfVarPerNoGood = NULL;
    //int* dev_prevLonelyVar = NULL;
    //int* dev_prevVarsAppearingInRemainingNoGoods=NULL;
    //int* dev_noGoodStatus=NULL; //the first column of the matrix is the status of the clause

    
    //allocates and copies the above arrays
    //storePrevStateOnDevice(/*data,&prevPartialAssignment, &prevNoOfVarPerNoGood, &prevLonelyVar, &noGoodStatus,&prevVarsAppearingInRemainingNoGoods*/);
    return false;
    //assigns the value to the variable
    //assignValueToVar(&data, var, value);
    
    /*
    pureLiteralCheck(&data);
    
    //nothing:
    learnClause();

    //if we find a conflict we backtrack (we need to revert the state first)
    if (unitPropagation(&data) == CONFLICT) {
        revert(&data, &prevPartialAssignment, &prevNoOfVarPerNoGood, &prevLonelyVar, &noGoodStatus,&prevVarsAppearingInRemainingNoGoods);
        return false;
    }
    //if the partialAssignment satisfies (falsifies) all the clauses we have found a solution
    if (data.currentNoGoods==0) {
        printf("SATISFIABLE\n");
        printf("Assignment:\n");
        printVarArray(data.partialAssignment);
        solutionFound=true;
        return true;
    }   
    //if there are no more variables to assign (AND having previously checked that not all the no good are sat) we backtrack
    if (data.varsYetToBeAssigned==0) {
        revert(&data, &prevPartialAssignment, &prevNoOfVarPerNoGood, &prevLonelyVar, &noGoodStatus,&prevVarsAppearingInRemainingNoGoods);
        return false;
    }
    //choose the next variable to assign
    int varToAssign = chooseVar(data.partialAssignment,data.varsAppearingInRemainingNoGoods);

    //the check is done just for reverting purposes in case we need to backtrack
    if ((solve(data, varToAssign, TRUE) || solve(data, varToAssign, FALSE)) == false) {
        revert(&data, &prevPartialAssignment, &prevNoOfVarPerNoGood, &prevLonelyVar, &noGoodStatus,&prevVarsAppearingInRemainingNoGoods);
        return false;
    }*/
    return true;
 }



//CREDITS TO: https://stackoverflow.com/questions/32530604/how-can-i-get-number-of-cores-in-cuda-device
//returns the number of cores per SM, used to optimize the number of threads per block
int getSMcores(struct cudaDeviceProp devProp){
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
        if (devProp.minor == 0) cores =  64;
        else if (devProp.minor == 6) cores =  128;
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
void printMatrix(int** matrix) {
    printf("\n");
    for (int i = 0; i < noNoGoods; i++) {
        if (matrix[i][0] == UNSATISFIED)
            printf("UNSATISFIED ");
        else
            printf("SATISFIED   ");
        printf("\n");
    }
    printf("\n");
}
void freeCUDA(){
    cudaFree(&dev_matrix);
    //TODO FINISH IT
}

void storePrevState(/*, int** dev_prevPartialAssignment, int** dev_prevNoOfVarPerNoGood, int** dev_prevLonelyVar, int** dev_noGoodStatus, int** dev_prevVarsAppearingInRemainingNoGoods*/) {
    /*
    (*prevPartialAssignment)=(int*)calloc(noVars + 1, sizeof(int));
    (*prevVarsAppearingInRemainingNoGoods)=(int*)calloc(noVars + 1, sizeof(int));
    (*prevNoOfVarPerNoGood) = (int*)calloc(noNoGoods, sizeof(int));
    (*prevLonelyVar) = (int*)calloc(noNoGoods, sizeof(int));
    (*noGoodStatus) = (int*)calloc(noNoGoods, sizeof(int));
    for (int i = 0; i < noVars + 1; i++) {
        (*prevPartialAssignment)[i] = data.partialAssignment[i];
        (*prevVarsAppearingInRemainingNoGoods)[i]= data.varsAppearingInRemainingNoGoods[i];
    }
    for (int i = 0; i < noNoGoods; i++) {
        (*prevNoOfVarPerNoGood)[i] = data.noOfVarPerNoGood[i];
        (*prevLonelyVar)[i] = data.lonelyVar[i];
        (*noGoodStatus)[i] = data.matrix[i][0];
    }*/
return;
}
