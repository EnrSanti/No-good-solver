//**THE CONSTANTS**//

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



//**DATA**//
//for the serial program
struct NoGoodData {
    int currentNoGoods; //the number of non satisfied clauses (yet)
    int** matrix; //the matrix that holds the clauses
    int* partialAssignment;//we skip the cell 0 in order to maintain coherence with the var numbering
    int* noOfVarPerNoGood; //a int array that holds the number of variables in each clause
    int* lonelyVar; //a int array that holds if noOfVarPerNoGood[i]==1 the index of the only variable in the clause
    int varsYetToBeAssigned; //the number of variables that are not yet assigned
    int *varsAppearingInRemainingNoGoods; //a int array keeping track for each variable in how many no goods it shows up
};

//for CUDA
struct NoGoodDataC {
    int currentNoGoods; //the number of non satisfied clauses (yet)
    int varsYetToBeAssigned; //the number of variables that are not yet assigned
    int* partialAssignment;//we skip the cell 0 in order to maintain coherence with the var numbering
    int* noOfVarPerNoGood; //a int array that holds the number of variables in each clause
    int* lonelyVar; //a int array that holds if noOfVarPerNoGood[i]==1 the index of the only variable in the clause  
};