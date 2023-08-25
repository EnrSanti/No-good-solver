#include <stdio.h>
#include <string.h>
#include <stdbool.h> 
#include <stdlib.h>

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

//for the literals in the partial assignment
#define UNASSIGNED 0
#define TRUE 1
#define FALSE -1

//for the pure literal check

#define FIRST_APPEARENCE 0
#define APPEARS_ONLY_POS 1
#define APPEARS_ONLY_NEG -1
#define APPEARS_BOTH 3

void readFile_allocateMatrix(const char *, struct NoGoodData*);
void printError(char *);
void popualteMatrix(FILE*, struct NoGoodData*);
void printMatrix(int **);
void printVarArray(int *);
void allocateMatrix(int ***);
void deallocateMatrix(int ***);
bool solve(struct NoGoodData,int,int);
int unitPropagation(struct NoGoodData* );
//void backJump();
void pureLiteralCheck(struct NoGoodData*);
void removeNoGoodSetsContaining(int***, int*,int,int);
int chooseVar();
void learnClause();
void assignValueToVar(struct NoGoodData*,int, int);
int removeLiteralFromNoGoods(struct NoGoodData* data,int, int);


int noVars=0; //the number of vars
int noNoGoods=0; //the no of clauses (initial)
int *varBothNegatedAndNot = NULL; //a int array that holds the status of the variables in the clauses (see the defines above)

struct NoGoodData {
    int currentNoGoods; //the number of non satisfied clauses (yet)
    int** matrix; //the matrix that holds the clauses
    int* partialAssignment;//we skip the cell 0 in order to maintain coherence with the var numbering
    int* noOfVarPerNoGood; //a int array that holds the number of variables in each clause
    int* lonelyVar; //a int array that holds if noOfVarPerNoGood[i]==1 the index of the only variable in the clause
};

void main(int argc, char const *argv[]){
	
	if(argc!=2){
		printError("Insert the file path");
		return;
	}

    struct NoGoodData data;
    readFile_allocateMatrix(argv[1],&data);

    printMatrix(data.matrix);
    printf("The status of the variables in the clauses: (%d doesn't appear, %d just positive, %d just negative, %d both)\n",FIRST_APPEARENCE, APPEARS_ONLY_POS,   APPEARS_ONLY_NEG,   APPEARS_BOTH);
    printVarArray(varBothNegatedAndNot);
    printf("\n");
	
    data.partialAssignment=(int *) calloc(noVars+1, sizeof(int));

	pureLiteralCheck(&data);
    
    if (unitPropagation(&data) == CONFLICT) {
        printf("UNSATISFIABLE\n");
        deallocateMatrix(&(data.matrix));
        return;
    }
    
    printMatrix(data.matrix);

    if(data.currentNoGoods == 0) {
		printf("SATISFIABLE\n");
		printf("Assignment:\n");
		printVarArray(data.partialAssignment);
        deallocateMatrix(&(data.matrix));
		return;
	}
    int varToAssign = chooseVar(data.partialAssignment);

    if (solve(data, varToAssign, TRUE) || solve(data, varToAssign, FALSE)) {
        printf("SATISFIABLE\n");
        printf("Assignment:\n");
        printVarArray(data.partialAssignment);
    }else {
        printf("UNSATISFIABLE\n");
    }
    deallocateMatrix(&(data.matrix));

}



//reads the content of a simil DMACS file (not the fanciest function but it's called just once)
void readFile_allocateMatrix(const char *str,struct NoGoodData* data){

	FILE* ptr;
    char ch;
    ptr = fopen(str, "r");
 
    if (NULL == ptr) {
        printError("No such file or can't be opened");
        return;
    }
 	bool isComment=true;
 	bool newLine=true;
    while (isComment==true && !feof(ptr)) {
        ch = fgetc(ptr);

        //a comment
        if(ch=='c' && newLine==true){
        	isComment=true;
        }
        if(ch=='p' && newLine==true){
        	isComment=false;
        }

        if(ch=='\n'){
        	newLine=true;
        }else{
        	newLine=false;
        }
    }

    //skip over "p nogood"
    int i=8;
    while(!feof(ptr) && i>0){
    	ch = fgetc(ptr);
    	i--;
    }

    //ignore return value for now
    fscanf (ptr, "%d", &noVars);      
    fscanf (ptr, "%d", &noNoGoods);

	printf("number of vars: %d \n",noVars);
    printf("number of nogoods: %d \n",noNoGoods);

    data->currentNoGoods=noNoGoods;

    popualteMatrix(ptr,data);
    
    fclose(ptr);
}

void popualteMatrix(FILE* ptr, struct NoGoodData* data){

	allocateMatrix(&(data->matrix));
    varBothNegatedAndNot = (int *)calloc(noVars + 1, sizeof(int));
    data->noOfVarPerNoGood = (int *)calloc(noNoGoods, sizeof(int));
    data->lonelyVar = (int *)calloc(noNoGoods, sizeof(int));

    for(int i = 0; i < noVars + 1; i++) {
		varBothNegatedAndNot[i] = FIRST_APPEARENCE;
	}

    int clauseCounter=0;
	int literal=0;

    while(!feof(ptr) && clauseCounter<noNoGoods){
		fscanf (ptr, "%d", &literal);
		if(literal==0){
			data->matrix[clauseCounter][0]=UNSATISFIED; //the first cell of the matrix is the status of the clause
			clauseCounter++;
		}else{
           
            int sign = literal > 0 ? POSITIVE_LIT : NEGATED_LIT;
            data->matrix[clauseCounter][literal*sign] = sign;
            data->noOfVarPerNoGood[clauseCounter]++;
            //if i have more vars i won't read this, so it can contain a wrong value
            data->lonelyVar[clauseCounter] = literal * sign;

            //populate the varBothNegatedAndNot array
            if(varBothNegatedAndNot[literal * sign]==FIRST_APPEARENCE)
				varBothNegatedAndNot[literal * sign]=sign;
            if(varBothNegatedAndNot[literal * sign]==APPEARS_ONLY_POS && sign==NEGATED_LIT)
                varBothNegatedAndNot[literal * sign]=APPEARS_BOTH;
			if(varBothNegatedAndNot[literal * sign]==APPEARS_ONLY_NEG && sign==POSITIVE_LIT)
				varBothNegatedAndNot[literal * sign]=APPEARS_BOTH;
		}
	}

	
}

//prints str with "ERROR" in front of it
void printError(char * str){
	printf("ERROR: %s \n",str);
}

void printMatrix(int ** matrix){
    printf("\n");
	for (int i = 0; i < noNoGoods; i++){
        if(matrix[i][0]==UNSATISFIED)
            printf("UNSATISFIED ");
        else
            printf("SATISFIED   ");
		for (int j = 1; j < noVars+1; j++){
			if(matrix[i][j]<0)
				printf("%d ", matrix[i][j]);
			else
				printf(" %d ", matrix[i][j]);
		}
		printf("\n");
	}
    printf("\n");
}

void allocateMatrix(int*** matrix){
	*matrix = (int **) calloc(noNoGoods, sizeof(int *));
	for (int i = 0; i < noNoGoods; i++){
 		(*matrix)[i] = (int *) calloc(noVars+1, sizeof(int));
	}

}

void deallocateMatrix(int ***matrix){

	for (int i = 0; i < noNoGoods; i++){
 		free((*matrix)[i]);
	}
	free((*matrix));
}

void printVarArray(int *array) {
    for (int i =1; i < noVars+1; i++) {
        printf("%d  ", array[i]);
    }
}

bool solve(struct NoGoodData data,int var, int value){

    assignValueToVar(&data ,var, value);
    pureLiteralCheck(&data);
    learnClause();
    if(unitPropagation(&data)==CONFLICT)
		return false;
    
    if (data.currentNoGoods==0) {
        printf("SATISFIABLE\n");
        printf("Assignment:\n");
        printVarArray(data.partialAssignment);
		return true;
    }
    int varToAssign = chooseVar(data.partialAssignment);
    return solve(data,varToAssign, TRUE) || solve(data, varToAssign, FALSE);
 }

int unitPropagation(struct NoGoodData* data){
    for (int i = 0; i < noNoGoods; i++) {
		if (data->noOfVarPerNoGood[i] == 1) {
            //lonelyVar[i] is a column index
            data->partialAssignment[data->lonelyVar[i]] = data->matrix[i][data->lonelyVar[i]] > 0 ? FALSE : TRUE;
            removeNoGoodSetsContaining(&(data->matrix), &(data->currentNoGoods),data->lonelyVar[i], (data->partialAssignment[data->lonelyVar[i]]) == TRUE ? NEGATED_LIT : POSITIVE_LIT);
            if (removeLiteralFromNoGoods(data, data->lonelyVar[i], data->partialAssignment[data->lonelyVar[i]] == TRUE ? POSITIVE_LIT : NEGATED_LIT)== CONFLICT) 
                return CONFLICT;
        }
	}
    return NO_CONFLICT;
}

void pureLiteralCheck(struct NoGoodData* data){
    for (int i = 1; i < noVars; i++) {
        if (varBothNegatedAndNot[i] == APPEARS_ONLY_POS) {
			data->partialAssignment[i] = FALSE;
            removeNoGoodSetsContaining(&(data->matrix), &(data->currentNoGoods),i,POSITIVE_LIT);
		} else if (varBothNegatedAndNot[i] == APPEARS_ONLY_NEG) {
            data->partialAssignment[i] = TRUE;
            removeNoGoodSetsContaining(&(data->matrix), &(data->currentNoGoods),i,NEGATED_LIT);
        }
    }
}

void removeNoGoodSetsContaining(int*** matrix,int *currentNoGoods,int varIndex,int sign) {

    //scan column (varIndex) of matrix
    for (int i = 0; i < noNoGoods; i++) {
		if ((*matrix)[i][varIndex] == sign) {
			//remove the nogood set
            (*matrix)[i][0] = SATISFIED;
			(*currentNoGoods)--;
		}
	}
}

int chooseVar(int * partialAssignment) {
    //return the fist unassigned var
    for(int i = 1; i < noVars; i++) {
		if (partialAssignment[i] == UNASSIGNED) {
			return i;
		}
	}
    //if all vars are assigned return -1 (never)
    return -1;
}
void assignValueToVar(struct NoGoodData* data, int varToAssign, int value) {
	data->partialAssignment[varToAssign] = value;
    removeNoGoodSetsContaining(&(data->matrix), &(data->currentNoGoods),varToAssign, value == TRUE ? NEGATED_LIT : POSITIVE_LIT);
}

void learnClause(){
    return;
    //TODO
}
int removeLiteralFromNoGoods(struct NoGoodData* data, int varIndex, int sign) {
	//scan column (varIndex) of matrix
	for (int i = 0; i < noNoGoods; i++) {
		if (data->matrix[i][varIndex] == sign) {
			//remove the literal
			data->matrix[i][varIndex] = 0;
			data->noOfVarPerNoGood[i]--;
			if (data->noOfVarPerNoGood[i] == 0) {
				return CONFLICT;
			}
		}
	}
	return NO_CONFLICT;
}