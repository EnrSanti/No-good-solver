//**THE CONSTANTS**//

//for the clauses
#define UNSATISFIED -2
#define SATISFIED 2

//for unit propagation
#define CONFLICT -3
#define NO_CONFLICT 3
#define RESET_CONFLCIT 0
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

int checkSolution(int** matrix,int* partialAssignment,int noNoGoods, int noVars){   
    int currentNGsat=0;   
    //for each no good we check if there's an assignment falsifying it
    for(int i=0; i<noNoGoods; i++){
        for(int j=1; j<noVars+1; j++){
            //can be optimized a lot, however is just to check
            if((matrix[i][j]==POSITIVE_LIT && partialAssignment[j]==FALSE) || (matrix[i][j]==NEGATED_LIT && partialAssignment[j]==TRUE)){
                currentNGsat=1;
                break;
            }
        }   
        if(currentNGsat==0){
            return 0;
        }else
            currentNGsat=0;
    }
    return 1;
}


//**DATA**//

//for the serial program
struct NoGoodData {
    int currentNoGoods; //the number of non satisfied clauses (yet)
    int** matrix; //the matrix that holds the clauses(only the fst col will be modified)
    int* partialAssignment;//we skip the cell 0 in order to maintain coherence with the var numbering
    int* noOfVarPerNoGood; //a int array that holds the number of variables in each clause
    int* lonelyVar; //a int array that holds if noOfVarPerNoGood[i]==1 the index of the only variable in the clause
    int varsYetToBeAssigned; //the number of variables that are not yet assigned
    int* varsAppearingInRemainingNoGoodsPositiveNegative;//a int array keeping track for each variable in how many no goods it shows up (positive and negative), the array has len 2*(vars+1)
};

//for CUDA (two almost identical structres, containing THE SAME data)
struct NoGoodDataCUDA_host {
    int currentNoGoods; //the number of non satisfied clauses (yet)
    int varsYetToBeAssigned; //the number of variables that are not yet assigned
    int* partialAssignment;//we skip the cell 0 in order to maintain coherence with the var numbering
    int* noOfVarPerNoGood; //a int array that holds the number of variables in each clause
    int* lonelyVar; //a int array that holds if noOfVarPerNoGood[i]==1 the index of the only variable in the clause  
    int* varsAppearingInRemainingNoGoodsPositiveNegative;//a int array keeping track for each variable in how many no goods it shows up (positive and negative), the array has len 2*(vars+1)
    int* matrix_noGoodsStatus; //the first column of the matrix
    int* unitPropValuestoRemove;//used to store on the signs (found by unit propagation) of the variables to eliminate, in the serial version this wasn't necessary since removeNoGoodSetsContaining was called for each variable inside of a loop in unitProp
};

struct NoGoodDataCUDA_devDynamic {
  
    int* dev_partialAssignment;//we skip the cell 0 in order to maintain coherence with the var numbering
    int* dev_noOfVarPerNoGood; //a int array that holds the number of variables in each clause
    int* dev_lonelyVar; //a int array that holds if noOfVarPerNoGood[i]==1 the index of the only variable in the clause  
    int* dev_varsAppearingInRemainingNoGoodsPositiveNegative;
    int* dev_matrix_noGoodsStatus; //the status of each clause (satisfied/unsatisfied) (used to avoid copying the whole matrix from device to host)
    int* dev_unitPropValuestoRemove; //used to store on the device the signs (found by unit propagation) of the variables to eliminate, in the serial version this wasn't necessary since removeNoGoodSetsContaining was called for each variable inside of a loop in unitProp
    
    //a piece of the former "NoGoodData", it contains two integer varaiables allocated on the device 
    int* dev_varsYetToBeAssigned_dev_currentNoGoods; //the number of variables that are not yet assigned and the number of non satisfied clauses (yet) 
};


/* APPARENTLY DIDN't WORK
//the following struct contains the data (static) for the device, the struct has to be declared __device__
__device__ struct NoGoodDataCUDA_devStatic {
    int dev_currentNoGoods; //the number of non satisfied clauses (yet)
    int dev_varsYetToBeAssigned; //the number of variables that are not yet assigned
};
*/