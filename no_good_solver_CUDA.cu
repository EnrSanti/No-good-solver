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
__global__ void pureLiteralCheck(int*, int *, int *, int *);
__global__ void removeNoGoodSetsContaining(int*, int*,int *,int *, int *);
__global__ void unitPropagation(int* ,int * ,int * , int * ,int* ,int *,int *);
__global__ void removeLiteralFromNoGoods(int*, int*,int*,int *,int* );
void printMatrix(int**);
int getSMcores(struct cudaDeviceProp);
bool solve(struct NoGoodDataCUDA_devDynamic,struct NoGoodDataCUDA_host,int,int);
int chooseVar(int *, int*);
void storePrevStateOnDevice(struct NoGoodDataCUDA_devDynamic, int**,int**,int**,int**,int**,int **);
void revert(struct NoGoodDataCUDA_devDynamic*, int**,int**,int**,int**,int**,int **);

//algorithm data:

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
int blocksToLaunch_VARS;
int blocksToLaunch_NG;

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
    blocksToLaunch_VARS = deviceProperties.multiProcessorCount;
    //if we have less variables than deviceProperties.multiProcessorCount*threadsPerBlock such that we can leave a SM empty and assigning just one var per thread we do so
    for(int i=1; i<=deviceProperties.multiProcessorCount; i++){
        if(i*threadsPerBlock>noVars){
            blocksToLaunch_VARS=i;
            break;
        }
    }

    //the same operation is done in order to get the blocks to launch for the kernels in which each thread deals with one or more no goods
    blocksToLaunch_NG= deviceProperties.multiProcessorCount;
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
    pureLiteralCheck<<<blocksToLaunch_VARS, threadsPerBlock>>>(dev_matrix,dev_data_dynamic.dev_partialAssignment, dev_varBothNegatedAndNot, dev_data_dynamic.dev_varsYetToBeAssigned_dev_currentNoGoods);
 	cudaDeviceSynchronize();
    //here threads deal with noGoods
    removeNoGoodSetsContaining<<<blocksToLaunch_NG, threadsPerBlock>>>(dev_matrix, dev_varBothNegatedAndNot,dev_varBothNegatedAndNot,dev_data_dynamic.dev_matrix_noGoodsStatus,(dev_data_dynamic.dev_varsYetToBeAssigned_dev_currentNoGoods+1));
    cudaDeviceSynchronize();
    //we do the unit propagation, and we subsequently remove the no goods and literals 
    unitPropagation<<<blocksToLaunch_NG, threadsPerBlock>>>(dev_matrix,dev_data_dynamic.dev_partialAssignment, dev_varBothNegatedAndNot,dev_data_dynamic.dev_noOfVarPerNoGood, dev_data_dynamic.dev_lonelyVar,dev_unitPropValuestoRemove,dev_data_dynamic.dev_varsYetToBeAssigned_dev_currentNoGoods);
    cudaDeviceSynchronize();
    removeNoGoodSetsContaining<<<blocksToLaunch_NG, threadsPerBlock>>>(dev_matrix, dev_varBothNegatedAndNot,dev_unitPropValuestoRemove, dev_data_dynamic.dev_matrix_noGoodsStatus,(dev_data_dynamic.dev_varsYetToBeAssigned_dev_currentNoGoods+1));
    cudaDeviceSynchronize();
    //here threads deal with noGoods
    removeLiteralFromNoGoods<<<blocksToLaunch_NG, threadsPerBlock>>>(dev_matrix,dev_data_dynamic.dev_noOfVarPerNoGood, dev_data_dynamic.dev_lonelyVar,dev_data_dynamic.dev_partialAssignment,dev_unitPropValuestoRemove);


    //to check whether a conflict occurred we get the data:
    cudaError_t err=cudaMemcpyFromSymbol(&conflict,dev_conflict, sizeof(int));
    //printf("copy conflict%s\n",cudaGetErrorString(err) );
    err=cudaMemcpy(&(data.currentNoGoods),dev_data_dynamic.dev_varsYetToBeAssigned_dev_currentNoGoods, sizeof(int),cudaMemcpyDeviceToHost);
    printf("vedi se errore: %s\n",cudaGetErrorString(err) );
   
    err=cudaMemcpy(&(data.varsYetToBeAssigned),(dev_data_dynamic.dev_varsYetToBeAssigned_dev_currentNoGoods+1), sizeof(int),cudaMemcpyDeviceToHost);
    printf("vedi se errore: %s\n",cudaGetErrorString(err) );
   
    cudaMemcpy(data.partialAssignment, dev_data_dynamic.dev_partialAssignment, (noVars+1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(data.matrix_noGoodsStatus, dev_data_dynamic.dev_matrix_noGoodsStatus, (noNoGoods) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(data.varsAppearingInRemainingNoGoods,dev_data_dynamic.dev_varsAppearingInRemainingNoGoods, (noVars + 1) * sizeof(int),cudaMemcpyDeviceToHost);

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

    for(int i=1; i<noVars+1; i++){
        printf("%d, ",data.partialAssignment[i]);   
    }

        printf("\n ");
    for(int i=1; i<noVars+1; i++){
        
        printf("%d, ",data.varsAppearingInRemainingNoGoods[i]);
    }
    //we choose a variable and we start the search
    int varToAssign = chooseVar(data.partialAssignment,data.varsAppearingInRemainingNoGoods);

    if (solve(dev_data_dynamic,data, varToAssign, TRUE) || solve(dev_data_dynamic,data, varToAssign, FALSE)) {
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

    err=cudaMalloc((void**)&(dev_data_dynamic->dev_varsYetToBeAssigned_dev_currentNoGoods), 2 * sizeof(int));   
    printf("VEDI SE DA ERRORE%s\n",cudaGetErrorString(err) );

    err=cudaMemcpy((dev_data_dynamic->dev_varsYetToBeAssigned_dev_currentNoGoods), &noNoGoods, sizeof(int), cudaMemcpyHostToDevice);
    printf("VEDI SE DA ERRORE%s\n",cudaGetErrorString(err) );
   
    err=cudaMemcpy((dev_data_dynamic->dev_varsYetToBeAssigned_dev_currentNoGoods+1), &noVars, sizeof(int), cudaMemcpyHostToDevice);
    printf("VEDI SE DA ERRORE%s\n",cudaGetErrorString(err) );
   
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
    cudaMemcpy((dev_data_dynamic->dev_varsAppearingInRemainingNoGoods), data->varsAppearingInRemainingNoGoods, sizeof(int) * (noVars + 1), cudaMemcpyHostToDevice);    
    
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
__global__ void  pureLiteralCheck(int* dev_matrix,int * dev_partialAssignment,int * dev_varBothNegatedAndNot, int* dev_varsYetToBeAssigned) {
 
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
        atomicAdd(dev_varsYetToBeAssigned,decrease);
}

//removes (assigns 'falsified' satisfied) the no goods if they contain the literal varIndex with the indicated sign

//one th per no good
__global__ void removeNoGoodSetsContaining(int* matrix, int* dev_varBothNegatedAndNot,int* sing,int* dev_matrix_noGoodsStatus, int* dev_currentNoGoods) {
	
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
        atomicAdd(dev_currentNoGoods, decrease);
}


__global__ void unitPropagation(int* dev_matrix,int * dev_partialAssignment,int * dev_varBothNegatedAndNot,int *dev_noOfVarPerNoGood,int* dev_lonelyVar, int* dev_unitPropValuestoRemove,int * dev_varsYetToBeAssigned) {
    
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
        atomicAdd((dev_varsYetToBeAssigned), decrease);

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

bool solve(struct NoGoodDataCUDA_devDynamic dev_data,struct NoGoodDataCUDA_host data, int var, int value) {
    
    //if we want to stop after the first solution and it's already found
    if (solutionFound && breakSearchAfterOne)
        return true;
    
    //local variables which will be used to revert the state of the data structure when backtracking
    int* dev_prevPartialAssignment = NULL;
    int* dev_prevNoOfVarPerNoGood = NULL;
    int* dev_prevLonelyVar = NULL;
    int* dev_prevVarsAppearingInRemainingNoGoods=NULL;
    int* dev_matrix_prevNoGoodsStatus=NULL; //the first column of the matrix is the status of the clause
    int* dev_prevVarsYetToBeAssigned_prevCurrentNoGoods; 

    //allocates and copies the above arrays
    storePrevStateOnDevice(dev_data,&dev_prevPartialAssignment, &dev_prevNoOfVarPerNoGood, &dev_prevLonelyVar, &dev_matrix_prevNoGoodsStatus,&dev_prevVarsAppearingInRemainingNoGoods,&dev_prevVarsYetToBeAssigned_prevCurrentNoGoods);

    //assigns the value to the variable (the former assignValueToVar method, unrolled for pointer issues)
    //we copy the value of the vatiable into the assignment
    //può essere fatto con un kernel
    
    cudaError_t err=cudaMemcpy((dev_data.dev_partialAssignment+var),&value,sizeof(int),cudaMemcpyHostToDevice);
    printf("copying in assignValueToVar: %s\n",cudaGetErrorString(err));
    
    err=cudaMemcpy(&(data.varsYetToBeAssigned),&(dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods[0]), sizeof(int),cudaMemcpyDeviceToHost);
    printf("copying in varsYetToBeAssigned: %s\n",cudaGetErrorString(err));
    data.varsYetToBeAssigned--;
    err=cudaMemcpy(&(dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods[0]),&(data.varsYetToBeAssigned), sizeof(int),cudaMemcpyHostToDevice);
    printf("copying in: %s\n",cudaGetErrorString(err));


    int* signs=(int*)calloc(noVars + 1, sizeof(int));
    signs[var]= (value == TRUE ? NEGATED_LIT : POSITIVE_LIT);
    int* dev_signs;
    err=cudaMalloc((void **)&dev_signs, (noVars + 1)* sizeof(int));
    printf("vedi: %s\n",cudaGetErrorString(err));
    removeNoGoodSetsContaining<<<blocksToLaunch_NG, threadsPerBlock>>>(dev_matrix, dev_varBothNegatedAndNot,dev_signs,dev_data.dev_matrix_noGoodsStatus,(dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods+1));
    cudaDeviceSynchronize();
    //end of former assignValueToVar
    

    //pure literal check
    pureLiteralCheck<<<blocksToLaunch_VARS, threadsPerBlock>>>(dev_matrix,dev_data.dev_partialAssignment, dev_varBothNegatedAndNot, dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods);
    cudaDeviceSynchronize();
    removeNoGoodSetsContaining<<<blocksToLaunch_NG, threadsPerBlock>>>(dev_matrix, dev_varBothNegatedAndNot,dev_varBothNegatedAndNot,dev_data.dev_matrix_noGoodsStatus,(dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods+1));
    cudaDeviceSynchronize();
    //end literal check
   
    //nothing:
    //learnClause();


    //unit propagation
    unitPropagation<<<blocksToLaunch_NG, threadsPerBlock>>>(dev_matrix,dev_data.dev_partialAssignment, dev_varBothNegatedAndNot,dev_data.dev_noOfVarPerNoGood, dev_data.dev_lonelyVar,dev_unitPropValuestoRemove,dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods);
    cudaDeviceSynchronize();
    removeNoGoodSetsContaining<<<blocksToLaunch_NG, threadsPerBlock>>>(dev_matrix, dev_varBothNegatedAndNot,dev_unitPropValuestoRemove, dev_data.dev_matrix_noGoodsStatus,(dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods+1));
    cudaDeviceSynchronize();
    removeLiteralFromNoGoods<<<blocksToLaunch_NG, threadsPerBlock>>>(dev_matrix,dev_data.dev_noOfVarPerNoGood, dev_data.dev_lonelyVar,dev_data.dev_partialAssignment,dev_unitPropValuestoRemove);
    //end unit propagation

     //to check whether a conflict occurred we get the data:
    err=cudaMemcpyFromSymbol(&conflict,dev_conflict, sizeof(int));
    printf("vedi: %s\n",cudaGetErrorString(err));
    

    if (conflict == CONFLICT) {
        revert(&dev_data, &dev_prevPartialAssignment, &dev_prevNoOfVarPerNoGood, &dev_prevLonelyVar, &dev_matrix_prevNoGoodsStatus,&dev_prevVarsAppearingInRemainingNoGoods,&dev_prevVarsYetToBeAssigned_prevCurrentNoGoods);
        return false;
    }
    err=cudaMemcpy(&(data.currentNoGoods),dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods, sizeof(int),cudaMemcpyDeviceToHost);
    printf("vedi se errore: %s\n",cudaGetErrorString(err) );

    //if the partialAssignment satisfies (falsifies) all the clauses we have found a solution
    if(data.currentNoGoods == 0) {
        printf("SATISFIABLE\n");
        printf("Assignment:\n");
        //printVarArray(data.partialAssignment);
        solutionFound=true;
        return true;
    }


    err=cudaMemcpy(&(data.varsYetToBeAssigned),&(dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods[0]), sizeof(int),cudaMemcpyDeviceToHost);
    printf("copying in varsYetToBeAssigned: %s\n",cudaGetErrorString(err));
  
    //if there are no more variables to assign (AND having previously checked that not all the no good are sat) we backtrack
    if (data.varsYetToBeAssigned==0) {
        revert(&dev_data, &dev_prevPartialAssignment, &dev_prevNoOfVarPerNoGood, &dev_prevLonelyVar, &dev_matrix_prevNoGoodsStatus,&dev_prevVarsAppearingInRemainingNoGoods,&dev_prevVarsYetToBeAssigned_prevCurrentNoGoods);
        return false;
    }
    

    cudaMemcpy(data.partialAssignment, dev_data.dev_partialAssignment, (noVars+1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(data.varsAppearingInRemainingNoGoods,dev_data.dev_varsAppearingInRemainingNoGoods, (noVars + 1) * sizeof(int),cudaMemcpyDeviceToHost);

    int varToAssign = chooseVar(data.partialAssignment,data.varsAppearingInRemainingNoGoods);
    
    //the check is done just for reverting purposes in case we need to backtrack
    if ((solve(dev_data,data, varToAssign, TRUE) || solve(dev_data,data, varToAssign, FALSE)) == false) {
        revert(&dev_data, &dev_prevPartialAssignment, &dev_prevNoOfVarPerNoGood, &dev_prevLonelyVar, &dev_matrix_prevNoGoodsStatus,&dev_prevVarsAppearingInRemainingNoGoods,&dev_prevVarsYetToBeAssigned_prevCurrentNoGoods);
        return false;
    }
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

void storePrevStateOnDevice(struct NoGoodDataCUDA_devDynamic dev_data, int** dev_prevPartialAssignment, int** dev_prevNoOfVarPerNoGood, int** dev_prevLonelyVar, int** dev_noGoodStatus, int** dev_prevVarsAppearingInRemainingNoGoods, int** dev_prevVarsYetToBeAssigned_prevCurrentNoGoods) {
    //we allocate the space on the global memory for the following arrays:
    cudaError_t err=cudaMalloc((void **)dev_prevPartialAssignment, (noVars + 1) * sizeof(int));
    printf("allocated varBothNegatedAndNot: %s\n",cudaGetErrorString(err));

    err=cudaMalloc((void **)dev_prevNoOfVarPerNoGood, noNoGoods* sizeof(int));
    printf("allocated dev_prevNoOfVarPerNoGood: %s\n",cudaGetErrorString(err));
    
    err=cudaMalloc((void **)dev_prevLonelyVar,noNoGoods * sizeof(int));
    printf("allocated varBothNegatedAndNot: %s\n",cudaGetErrorString(err) );

    err=cudaMalloc((void **)dev_noGoodStatus, noNoGoods * sizeof(int));
    printf("allocated dev_noGoodStatus: %s\n",cudaGetErrorString(err) );

    err=cudaMalloc((void **)dev_prevVarsAppearingInRemainingNoGoods, (noVars + 1) * sizeof(int));
    printf("allocated dev_prevVarsAppearingInRemainingNoGoods: %s\n",cudaGetErrorString(err) );

    //we also copy the two (global) variables in a [2] array, we can't declare a device var locally ( we could use a kernel 1x1, maybe it would be faster (TODO))
    err=cudaMalloc((void **)&dev_prevVarsYetToBeAssigned_prevCurrentNoGoods, 2 * sizeof(int));
    printf("copying in dev_prevVarsYetToBeAssigned_prevCurrentNoGoods: %s\n",cudaGetErrorString(err));
    


    //we copy the contents
    //according to: https://stackoverflow.com/questions/6063619/cuda-device-to-device-transfer-expensive to optimize the copy we could make another kernel
    err=cudaMemcpy((*dev_prevPartialAssignment), dev_data.dev_partialAssignment,sizeof(int)*(noVars + 1) , cudaMemcpyDeviceToDevice);
    printf("copying: %s\n",cudaGetErrorString(err));

    err=cudaMemcpy((*dev_prevNoOfVarPerNoGood), dev_data.dev_noOfVarPerNoGood,sizeof(int)*(noVars + 1) , cudaMemcpyDeviceToDevice);
    printf("copying: %s\n",cudaGetErrorString(err));

    err=cudaMemcpy((*dev_prevLonelyVar), dev_data.dev_lonelyVar,sizeof(int)*(noVars + 1) , cudaMemcpyDeviceToDevice);
    printf("copying: %s\n",cudaGetErrorString(err));

    err=cudaMemcpy((*dev_noGoodStatus), dev_data.dev_matrix_noGoodsStatus,sizeof(int)*(noVars + 1) , cudaMemcpyDeviceToDevice);
    printf("copying: %s\n",cudaGetErrorString(err));

    err=cudaMemcpy((*dev_prevVarsAppearingInRemainingNoGoods), dev_data.dev_varsAppearingInRemainingNoGoods,sizeof(int)*(noVars + 1) , cudaMemcpyDeviceToDevice);
    printf("copying: %s\n",cudaGetErrorString(err));

    err=cudaMemcpy(dev_prevVarsYetToBeAssigned_prevCurrentNoGoods, (dev_data.dev_varsYetToBeAssigned_dev_currentNoGoods) ,sizeof(int)*2, cudaMemcpyDeviceToDevice);
    printf("copying in dev_prevVarsYetToBeAssigned_prevCurrentNoGoods: %s\n",cudaGetErrorString(err));

}

//performs a copy of the arrays passed (to revert to the previous state) then it deallocates the memory
void revert(struct NoGoodDataCUDA_devDynamic* dev_data, int** dev_prevPartialAssignment, int** dev_prevNoOfVarPerNoGood, int** dev_prevLonelyVar, int** dev_noGoodStatus, int** dev_prevVarsAppearingInRemainingNoGoods,int ** dev_prevVarsYetToBeAssigned_prevCurrentNoGoods) {
    cudaError_t err=cudaMemcpy((dev_data->dev_partialAssignment),(*dev_prevPartialAssignment),sizeof(int)*(noVars + 1) , cudaMemcpyDeviceToDevice);
    printf("copying: %s\n",cudaGetErrorString(err));

    err=cudaMemcpy((dev_data->dev_noOfVarPerNoGood),(*dev_prevNoOfVarPerNoGood),sizeof(int)*(noVars + 1) , cudaMemcpyDeviceToDevice);
    printf("copying: %s\n",cudaGetErrorString(err));

    err=cudaMemcpy((dev_data->dev_lonelyVar),(*dev_prevLonelyVar),sizeof(int)*(noVars + 1) , cudaMemcpyDeviceToDevice);
    printf("copying: %s\n",cudaGetErrorString(err));

    err=cudaMemcpy((dev_data->dev_matrix_noGoodsStatus),(*dev_noGoodStatus),sizeof(int)*(noVars + 1) , cudaMemcpyDeviceToDevice);
    printf("copying: %s\n",cudaGetErrorString(err));

    err=cudaMemcpy((dev_data->dev_varsAppearingInRemainingNoGoods),(*dev_prevVarsAppearingInRemainingNoGoods),sizeof(int)*(noVars + 1) , cudaMemcpyDeviceToDevice);
    printf("copying: %s\n",cudaGetErrorString(err));

    err=cudaMemcpy((dev_data->dev_varsYetToBeAssigned_dev_currentNoGoods),(*dev_prevVarsYetToBeAssigned_prevCurrentNoGoods),sizeof(int)*2, cudaMemcpyDeviceToDevice);
    printf("VEDIIIIIIIIIIIII: %s\n",cudaGetErrorString(err));
    
    cudaFree(*dev_prevPartialAssignment);
    cudaFree(*dev_prevNoOfVarPerNoGood);
    cudaFree(*dev_prevLonelyVar);
    cudaFree(*dev_prevVarsAppearingInRemainingNoGoods);
    cudaFree(*dev_noGoodStatus);
    cudaFree(*dev_prevVarsYetToBeAssigned_prevCurrentNoGoods);
}
