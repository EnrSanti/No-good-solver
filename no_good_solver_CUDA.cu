//THE FOLLOWING PROGRAM in the current version can support only the use of one GPU
#include <stdio.h>
#include <string.h>
#include <stdbool.h> 
#include <stdlib.h>
#include <math.h>

//apparently this is needed for cuda ON WINDOWS
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <driver_types.h>

//for the clauses
#define UNSATISFIED -2
#define SATISFIED 2

//for unit propagation
#define CONFLICT -3
#define NO_CONFLICT 3

//for the problem
#define SATISFIABLE true
#define UNSATISFIABLE false

//for the literals in the matrix
#define NEGATED_LIT -1
#define POSITIVE_LIT 1
#define NO_LIT 0
//for the literals in the partial assignment
#define UNASSIGNED 0
#define TRUE 1
#define FALSE -1

//for the pure literal check
#define FIRST_APPEARENCE 0
#define APPEARS_ONLY_POS 1
#define APPEARS_ONLY_NEG -1
#define APPEARS_BOTH 3


struct NoGoodDataC {
    int currentNoGoods; //the number of non satisfied clauses (yet)
    int varsYetToBeAssigned; //the number of variables that are not yet assigned
    int* partialAssignment;//we skip the cell 0 in order to maintain coherence with the var numbering
    int* noOfVarPerNoGood; //a int array that holds the number of variables in each clause
    int* lonelyVar; //a int array that holds if noOfVarPerNoGood[i]==1 the index of the only variable in the clause  
};

void readFile_allocateMatrix(const char*, struct NoGoodDataC*);
void printError(char*);
void popualteMatrix(FILE*, struct NoGoodDataC*);
void allocateMatrix();
void deallocateMatrix();
__global__ void pureLiteralCheck(int*, int *, int *);
__global__ void removeNoGoodSetsContaining(int*, int*, int*,int *);
__global__ void unitPropagation(int* ,int * ,int * , int * ,int* ,int *);
int removeLiteralFromNoGoods(int**, int*,int*,int *,int, int);
void printMatrix(int**);
int getSMcores(struct cudaDeviceProp);

//algorithm data:

int** matrix; //the matrix that holds the clauses
int* dev_matrix; //the matrix that holds the clauses on the device
int* matrix_noGoodsStatus;
int* dev_matrix_noGoodsStatus; //the status of each clause (satisfied/unsatisfied) (used to avoid copying the whole matrix from device to host)


int noVars = -1; //the number of vars in the model
__device__ int dev_noVars; //the number of vars in the model on the device

int noNoGoods = -1; //the no of clauses (initially) in the model
__device__ int dev_noNoGoods; //the no of clauses (initially) in the model on the device

//create the strucure
struct NoGoodDataC data;
__device__ int dev_currentNoGoods; //the number of non satisfied clauses (yet)
__device__ int dev_varsYetToBeAssigned; //the number of variables that are not yet assigned
int* dev_partialAssignment;//we skip the cell 0 in order to maintain coherence with the var numbering
int* dev_noOfVarPerNoGood; //a int array that holds the number of variables in each clause
int* dev_lonelyVar; //a int array that holds if noOfVarPerNoGood[i]==1 the index of the only variable in the clause  

int* varBothNegatedAndNot = NULL; //a int array that holds the status of the variables in the clauses (see the defines above)
int* dev_varBothNegatedAndNot=NULL;
bool breakSearchAfterOne = false; //if true, the search will stop after the first solution is found
bool solutionFound = false; //if true, a solution was found, used to stop the search
int conflict=NO_CONFLICT;

//technical (GPU related) data:
struct cudaDeviceProp deviceProperties; //on WINDOWS it seems we need to add the "Struct" 
int noOfVarsPerThread; //the number of variables that each thread will handle in unit propagation, so that each thread will deal with 32 byte of memory (1 mem. transfer)
int noNoGoodsperThread;
int threadsPerBlock; //the number of threads per block, we want to have the maximum no of warps  to utilize the full SM


//device auxiliary variables
__device__ int dev_noOfVarsPerThread;
__device__ int dev_noNoGoodsperThread;
int* dev_conflict; //used to store whether the propagation caused a conflict
int* dev_unitPropValuestoRemove; 

int main(int argc, char const* argv[]) {

    int GPUSno;

    //we just check, then GPUSno won't be used to scale the program
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
    printf("The detected GPU has %d SMs \n", deviceProperties.multiProcessorCount);
    //we check if the GPU is supported, since we will launch at least 64 threads per block (in order to have 4 warps per block, and exploit the new GPUs)
    if (deviceProperties.maxThreadsDim[0] < 64 || getSMcores(deviceProperties)==0) {
        //printError("The GPU is not supported, it has less than 64 threads per block/UNKNOWN DEVICE TYPE, buy a better one :)");
		return -3;
    }

    //we populate it with the data from the file
    readFile_allocateMatrix(argv[1], &data);

    //we want at least one block per SM (in order to use the full capacity of the GPU)
    int blocksToLaunch = deviceProperties.multiProcessorCount;
    //we get how many varibles and no goods each thread will handle 
    threadsPerBlock = getSMcores(deviceProperties);
    noOfVarsPerThread = ceil((float) (noVars / (threadsPerBlock * deviceProperties.multiProcessorCount)));
    noNoGoodsperThread = ceil((float) (noNoGoods / (threadsPerBlock * deviceProperties.multiProcessorCount)));

    printf("No of vars per th. %d \n", noOfVarsPerThread);
    printf("No of no goods per th %d \n", noNoGoodsperThread);

    //we copy the data to the device
    cudaMemcpyToSymbol(dev_noOfVarsPerThread, &noOfVarsPerThread, sizeof(int), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(dev_noNoGoodsperThread, &noNoGoodsperThread, sizeof(int), 0, cudaMemcpyHostToDevice);

    //thus we launch the number of blocks needed, each thread will handle noOfVarsPerThread variables (on newer GPUS 128 threads per block, four warps)

    //we launch the kernel that will handle the pure literals
    //here threads deal with vars
    pureLiteralCheck<<<blocksToLaunch, threadsPerBlock>>>(dev_matrix,dev_partialAssignment, dev_varBothNegatedAndNot);
 	
    //here threads deal with noGoods
    removeNoGoodSetsContaining<<<blocksToLaunch, threadsPerBlock>>>(dev_matrix,dev_partialAssignment, dev_varBothNegatedAndNot,dev_varBothNegatedAndNot);
    printMatrix(matrix);

    conflict=NO_CONFLICT;

    unitPropagation<<<blocksToLaunch, threadsPerBlock >>>(dev_matrix,dev_partialAssignment, dev_varBothNegatedAndNot,dev_noOfVarPerNoGood, dev_lonelyVar,dev_unitPropValuestoRemove);
    removeNoGoodSetsContaining<<<blocksToLaunch, threadsPerBlock >>>(dev_matrix,dev_partialAssignment, dev_varBothNegatedAndNot,dev_unitPropValuestoRemove);
    

    //we copy the matrix (just the first column) back on the host
    for (int i = 0; i < noNoGoods; i++) {
        cudaMemcpy(matrix[i], dev_matrix + i * ((noVars + 1)), (noVars + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    }

    //we copy noOfVarPerNoGood, lonelyVar e partialAssignment back on the host
    cudaMemcpy(data.noOfVarPerNoGood, dev_noOfVarPerNoGood, (noNoGoods) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(data.lonelyVar, dev_lonelyVar, (noNoGoods) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(data.partialAssignment, dev_partialAssignment, (noVars+1) * sizeof(int), cudaMemcpyDeviceToHost);
    
    

    for(int i=0; i<noNoGoods; i++){
        if (matrix[i][0] ==  UNSATISFIED &&  data.noOfVarPerNoGood[i] == 1) 
            conflict=removeLiteralFromNoGoods(matrix,data.noOfVarPerNoGood,data.lonelyVar,data.partialAssignment,data.lonelyVar[i], data.partialAssignment[data.lonelyVar[i]] == TRUE ? POSITIVE_LIT : NEGATED_LIT);
        //if we find a conlfict at the top level, the problem is unsatisfiable
        if(conflict==CONFLICT)
            break;
    } 
    printMatrix(matrix);
    if (conflict==CONFLICT) {
        printf("\n\n\n**********UNSATISFIABLE**********\n\n\n");
        //deallocateMatrix(&(data.matrix));
        return -1;
    }

    return 1;
}

//reads the content of a simil DMACS file and populates the data structure
// (not the fanciest function but it's called just once)
void readFile_allocateMatrix(const char* str, struct NoGoodDataC* data) {

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

    cudaError_t err=cudaMemcpyToSymbol(dev_noNoGoods, &noNoGoods, sizeof(int), 0, cudaMemcpyHostToDevice);
    //printf("copy No goods%s\n",cudaGetErrorString(err) );
    err=cudaMemcpyToSymbol(dev_noVars, &noVars, sizeof(int), 0, cudaMemcpyHostToDevice);
	//printf("copy No vars%s\n",cudaGetErrorString(err) );

    data->currentNoGoods = noNoGoods;
    data->varsYetToBeAssigned = noVars;   

    err=cudaMemcpyToSymbol(dev_currentNoGoods, &noNoGoods, sizeof(int), 0, cudaMemcpyHostToDevice);
    //printf("copy current No goods%s\n",cudaGetErrorString(err) );
    err=cudaMemcpyToSymbol(dev_varsYetToBeAssigned, &noVars, sizeof(int), 0, cudaMemcpyHostToDevice);
    //printf("copy current No vars%s\n",cudaGetErrorString(err) );
    popualteMatrix(ptr, data);
    fclose(ptr);
}


//subprocedure called by readFile_allocateMatrix it populates the data structure and other arrays such as varBothNegatedAndNot
void popualteMatrix(FILE* ptr, struct NoGoodDataC* data) {

    allocateMatrix();

    varBothNegatedAndNot = (int*)calloc(noVars + 1, sizeof(int));
    cudaError_t err=cudaMalloc((void **)&dev_varBothNegatedAndNot, (noVars + 1) * sizeof(int));
	//printf("allocated varBothNegatedAndNot %s\n",cudaGetErrorString(err) );
    data->noOfVarPerNoGood = (int*)calloc(noNoGoods, sizeof(int));
    data->lonelyVar = (int*)calloc(noNoGoods, sizeof(int));
    data->partialAssignment = (int*)calloc(noVars + 1, sizeof(int));
    matrix_noGoodsStatus= (int*)calloc(noNoGoods, sizeof(int));

    err=cudaMalloc((void**)&dev_partialAssignment, (noVars + 1) * sizeof(int));
    //printf("allocated dev_partialAssignment %s\n",cudaGetErrorString(err) );
    err=cudaMalloc((void**)&dev_noOfVarPerNoGood, noNoGoods* sizeof(int));
	//printf("allocated dev_noOfVarPerNoGood %s\n",cudaGetErrorString(err) );
    err=cudaMalloc((void**)&dev_matrix_noGoodsStatus, noNoGoods * sizeof(int));
    printf("allocated dev_noOfVarPerNoGood %s\n", cudaGetErrorString(err));
    err=cudaMalloc((void**)&dev_lonelyVar, noNoGoods * sizeof(int));
    err=cudaMalloc((void**)&dev_unitPropValuestoRemove, (noVars + 1) * sizeof(int));
	printf("allocated dev_lonelyVar %s\n",cudaGetErrorString(err) );


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

            //populate the varBothNegatedAndNot array
            if (varBothNegatedAndNot[literal * sign] == FIRST_APPEARENCE)
                varBothNegatedAndNot[literal * sign] = sign;
            if (varBothNegatedAndNot[literal * sign] == APPEARS_ONLY_POS && sign == NEGATED_LIT)
                varBothNegatedAndNot[literal * sign] = APPEARS_BOTH;
            if (varBothNegatedAndNot[literal * sign] == APPEARS_ONLY_NEG && sign == POSITIVE_LIT)
                varBothNegatedAndNot[literal * sign] = APPEARS_BOTH;
        }
    }
    //we now copy the content of the matrix to the device (https://forums.developer.nvidia.com/t/passing-dynamically-allocated-2d-array-to-device/43727 apparenlty works just for static matrices)
 	for(int i = 0; i < noNoGoods; i++) {
		cudaError_t err= cudaMemcpy((dev_matrix+i* ((noVars + 1) )), matrix[i], (noVars + 1) * sizeof(int), cudaMemcpyHostToDevice);
        matrix_noGoodsStatus[i] = UNSATISFIED;
    }

    //we copy varBothNegatedAndNot
    cudaMemcpy(dev_varBothNegatedAndNot , varBothNegatedAndNot, sizeof(int)* (noVars + 1), cudaMemcpyHostToDevice);
  	
    //printf("copied dev_varBothNegatedAndNot %s\n",cudaGetErrorString(err) );
    cudaMemcpy(dev_partialAssignment, (data->partialAssignment), sizeof(int) * (noVars + 1), cudaMemcpyHostToDevice);
   
    
    cudaMemcpy(dev_noOfVarPerNoGood, data->noOfVarPerNoGood, sizeof(int) * (noNoGoods), cudaMemcpyHostToDevice);
    err=cudaMemcpy(dev_matrix_noGoodsStatus, matrix_noGoodsStatus, sizeof(int) * (noNoGoods), cudaMemcpyHostToDevice);
    printf("copied dev_matrix_noGoodsStatus %s\n",cudaGetErrorString(err) );
    cudaMemcpy(dev_lonelyVar, data->lonelyVar, sizeof(int) * (noNoGoods), cudaMemcpyHostToDevice);
    

}

//prints str with "ERROR" in front of it
void printError(char* str) {
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
    cudaFree(&dev_matrix);
}

//removes the literal (by assigning a value) from the no goods IF it's UNASSIGNED and shows up with only one sign (in the remaining no goods)
//one th per var
__global__ void  pureLiteralCheck(int* dev_matrix,int * dev_partialAssignment,int * dev_varBothNegatedAndNot) {
 
    int thPos = blockIdx.x * blockDim.x + threadIdx.x;
    //printf("Th. %d\n",thPos);
    //we scan each var (ths deal with vars)
    for (int i = thPos*dev_noOfVarsPerThread; i < thPos*dev_noOfVarsPerThread + dev_noOfVarsPerThread; i++) {
    	if (i<=dev_noVars && dev_partialAssignment[i] == UNASSIGNED && (dev_varBothNegatedAndNot[i] == APPEARS_ONLY_POS || dev_varBothNegatedAndNot[i] == APPEARS_ONLY_NEG)) {
            printf("Th. %d operating on var: %d \n", thPos,i);
	    	//printf("th. no %d working on pos %d \n",thPos,i);
	        dev_partialAssignment[i] = -dev_varBothNegatedAndNot[i];
	        //printf("th. no %d assigning to var %d\n",thPos,dev_varBothNegatedAndNot[i]);
	        //TODO substiture with one decrement at the end (e.g. warp level reduction)
	        atomicAdd(&dev_varsYetToBeAssigned,-1);
	        
	        //this can't be called here, it would need too much serialization
	        //removeNoGoodSetsContaining(i,&(dev_matrix), &(dev_currentNoGoods), dev_varBothNegatedAndNot[i]);
    	}
    	__syncthreads();
    }   
   // }
}

//removes (assigns 'falsified' satisfied) the no goods if they contain the literal varIndex with the indicated sign

//one th per no good
__global__ void removeNoGoodSetsContaining(int* matrix, int* currentNoGoods, int* dev_varBothNegatedAndNot,int* sing) {
	
	int thPos = blockIdx.x * blockDim.x + threadIdx.x;
  
	//printf("th %d dev_noNoGoods %d \n",thPos,dev_noNoGoods);         
    //we scan each no_good (each th scans reading the first cell of matrix and the relative var pos)
    
    int decrease=0;
    for (int i = thPos*dev_noNoGoodsperThread; i <= thPos*dev_noNoGoodsperThread + dev_noNoGoodsperThread; i++) {
	    if (i<dev_noNoGoods){
	    	//fixed a no good (thread) we loop on the row of the matrix (in this way ONLY ONE thead access each cell of the first column)
	    	for(int varIndex=1; varIndex<=dev_noVars; varIndex++){

		    	if(sing[varIndex] !=0 && *(matrix+i*(dev_noVars+1)+varIndex) == sing[varIndex] && (*matrix + i*(dev_noVars+1)) != SATISFIED) {
			        //remove the nogood set
			        
                    *(matrix + i*(dev_noVars+1)) = SATISFIED; //VA FATTA ATOMIC
			      
			        //TODO substiture with one decrement at the end (e.g. warp level reduction)
			       	decrease--;
			        //(*currentNoGoods)--;
			    }
			    __syncthreads();
		    }
		}
    }
    atomicAdd(currentNoGoods, decrease);
}
//prints the content of the matrix (the first column is the status of each clause)
void printMatrix(int** matrix) {
    printf("\n");
    for (int i = 0; i < noNoGoods; i++) {
        if (matrix[i][0] == UNSATISFIED)
            printf("UNSATISFIED ");
        else
            printf("SATISFIED   ");
        for (int j = 1; j < noVars + 1; j++) {
            if (matrix[i][j] < 0)
                printf("%d ", matrix[i][j]);
            else
                printf(" %d ", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

__global__ void unitPropagation(int* dev_matrix,int * dev_partialAssignment,int * dev_varBothNegatedAndNot,int *dev_noOfVarPerNoGood,int* dev_lonelyVar, int* dev_unitPropValuestoRemove) {
    
    int thPos = blockIdx.x * blockDim.x + threadIdx.x;
    //int removeLiteralFromNoGoodsRETURN;
    int i;
    //for each no good (th deals with ng)
    for (int c = 0; c < dev_noNoGoodsperThread; c++) {
        i=thPos+32*c;
        //printf("UNIT PROP: th %d at no NOOD: %d\n",thPos,i);
    	//printf("th %d lookin at cell %d \n",thPos,i );
    	if(i<dev_noNoGoods && *(dev_matrix+i*(dev_noVars+1)) == UNSATISFIED && dev_noOfVarPerNoGood[i] == 1){
            //printf("UNIT PROP: at no good: %d, seing: %d (-2 = UNSATISFIED)\n",i,*(dev_matrix+i*(dev_noVars+1)));
            //lonelyVar[i] is a column index

            dev_partialAssignment[dev_lonelyVar[i]] = *(dev_matrix+i*(dev_noVars+1)+dev_lonelyVar[i]) > 0 ? FALSE : TRUE;
            atomicAdd(&dev_varsYetToBeAssigned,-1);
            dev_unitPropValuestoRemove[dev_lonelyVar[i]]=dev_partialAssignment[dev_lonelyVar[i]] == TRUE ? NEGATED_LIT : POSITIVE_LIT;

        }
        __syncthreads();

    }

}
//removes the literal varIndex from the nogood if the sign is the one indicated
int removeLiteralFromNoGoods(int** matrix, int* noOfVarPerNoGood,int* lonelyVar,int *partialAssignment,int varIndex, int sign) {
    //scan column (varIndex) of matrix
    for (int i = 0; i < noNoGoods; i++) {
        if (matrix[i][varIndex] == sign) {
            
            //data->matrix[i][varIndex] = 0; //not necessary WE NEVER MODIFY MATRIX (except for the first col)
            
            //remove the literal
            noOfVarPerNoGood[i]--;
            if(noOfVarPerNoGood[i]==1){
                //search and assing the literal to the lonelyVar
                for (int j = 1; j < noVars + 1; j++) {
                    if (matrix[i][j] != NO_LIT && partialAssignment[j]==UNASSIGNED) {
                        lonelyVar[i] = j;
                    }
                }
            }
            if (noOfVarPerNoGood[i] == 0) {
                return CONFLICT;
            }
        }
    }
    return NO_CONFLICT;
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