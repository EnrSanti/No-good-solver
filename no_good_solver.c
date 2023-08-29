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

struct NoGoodData {
    int currentNoGoods; //the number of non satisfied clauses (yet)
    int** matrix; //the matrix that holds the clauses
    int* partialAssignment;//we skip the cell 0 in order to maintain coherence with the var numbering
    int* noOfVarPerNoGood; //a int array that holds the number of variables in each clause
    int* lonelyVar; //a int array that holds if noOfVarPerNoGood[i]==1 the index of the only variable in the clause
    int varsYetToBeAssigned; //the number of variables that are not yet assigned
};

void readFile_allocateMatrix(const char *, struct NoGoodData*);
void printError(char *);
void popualteMatrix(FILE*, struct NoGoodData*);
void printMatrix(int**);
void printVarArray(int *);
void allocateMatrix(int ***);
void deallocateMatrix(int***);
bool solve(struct NoGoodData,int,int);
int unitPropagation(struct NoGoodData* );
void pureLiteralCheck(struct NoGoodData*);
void removeNoGoodSetsContaining(int***, int*,int,int);
int chooseVar();
void learnClause();
void assignValueToVar(struct NoGoodData*,int, int);
int removeLiteralFromNoGoods(struct NoGoodData* ,int, int);
void storePrevState(struct NoGoodData, int**, int**, int**, int**);
void revert(struct NoGoodData*, int**, int**, int**, int**);
//void backJump();

int noVars=0; //the number of vars in the model
int noNoGoods=0; //the no of clauses (initially) in the model
int *varBothNegatedAndNot = NULL; //a int array that holds the status of the variables in the clauses (see the defines above)
bool breakSearchAfterOne=false; //if true, the search will stop after the first solution is found
bool solutionFound = false; //if true, a solution was found, used to stop the search

void main(int argc, char const *argv[]){
	
    //if the user didn't insert the file path or typed more
	if(argc!=2){
		printError("Insert the file path");
		return;
	}
    //argv[1] = "testsNG\\test_3.txt"; just for testing

    //create the strucure
    struct NoGoodData data;
    //we populate it with the data from the file
    readFile_allocateMatrix(argv[1],&data);
    //print the matrix
    printMatrix(data.matrix);
    printf("\n");
	
    //allocate the partial assignment array
    data.partialAssignment=(int *) calloc(noVars+1, sizeof(int));


	pureLiteralCheck(&data);
    
    //if we find a conlfict at the top level, the problem is unsatisfiable
    if (unitPropagation(&data) == CONFLICT) {
        printf("\n\n\n**********UNSATISFIABLE**********\n\n\n");
        deallocateMatrix(&(data.matrix));
        return;
    }
    
    //if we somehow already have an assignment, we can skip the search
    if(data.currentNoGoods == 0) {
        printf("\n\n\n**********SATISFIABLE**********\n\n\n");
		printf("Assignment:\n");
		printVarArray(data.partialAssignment);
        deallocateMatrix(&(data.matrix));
		return;
	}
    
    //we choose a variable and we start the search
    int varToAssign = chooseVar(data.partialAssignment);

    if (solve(data, varToAssign, TRUE) || solve(data, varToAssign, FALSE)) {
        printf("\n\n\n**********SATISFIABLE**********\n\n\n");
    }else {
        printf("\n\n\n**********UNSATISFIABLE**********\n\n\n");
    }

    deallocateMatrix(&(data.matrix));

}



//reads the content of a simil DMACS file and populates the data structure
// (not the fanciest function but it's called just once)
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
    data->varsYetToBeAssigned=noVars;

    popualteMatrix(ptr,data);
    
    fclose(ptr);
}

//subprocedure called by readFile_allocateMatrix it populates the data structure and other arrays such as varBothNegatedAndNot
void popualteMatrix(FILE* ptr, struct NoGoodData* data){

	allocateMatrix(&(data->matrix));
    varBothNegatedAndNot = (int *)calloc(noVars + 1, sizeof(int));
    data->noOfVarPerNoGood = (int *)calloc(noNoGoods, sizeof(int));
    data->lonelyVar = (int *)calloc(noNoGoods, sizeof(int));

    for(int i = 0; i < noVars + 1; i++) {
		varBothNegatedAndNot[i] = FIRST_APPEARENCE;
	}

    int clauseCounter=0;
    int literal = 0;
    while(!feof(ptr) && clauseCounter<noNoGoods){
       
        //no idea why fscanf READS positive number as negative and vv (on Windows) 
        fscanf(ptr, "%d", &literal);
		if(literal==0){
			data->matrix[clauseCounter][0]=UNSATISFIED; //the first cell of the matrix is the status of the clause
			clauseCounter++;
		}else{
           
            int sign = literal > 0 ? POSITIVE_LIT : NEGATED_LIT;
            data->matrix[clauseCounter][literal*sign] = sign;
            data->noOfVarPerNoGood[clauseCounter]++;
            //if i have more vars i won't read this, so it can contain a wrong value (if the literal is just one the value will be correct)
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

//prints the content of the matrix (the first column is the status of each clause)
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

//allocates the matrix
void allocateMatrix(int*** matrix){
	*matrix = (int **) calloc(noNoGoods, sizeof(int *));
	for (int i = 0; i < noNoGoods; i++){
 		(*matrix)[i] = (int *) calloc(noVars+1, sizeof(int));
	}

}

//deallocates the matrix
void deallocateMatrix(int ***matrix){

	for (int i = 0; i < noNoGoods; i++){
 		free((*matrix)[i]);
	}
	free((*matrix));
}

//prints the content of the array passed (which for simplicity is an array of len noVars+1)
//the method is not general on purpose
void printVarArray(int *array) {
    for (int i =1; i < noVars+1; i++) {
        printf("%d  ", array[i]);
    }
}

//the main method responsible for calling unitpropagation, pureLiteralCheck and assignValueToVar
//performs the search by recursively calling itself twice (once for each value of the variable)
bool solve(struct NoGoodData data, int var, int value) {

    //if we want to stop after the first solution and it's already found
    if (solutionFound && breakSearchAfterOne)
        return true;
    
    //local variables which will be used to revert the state of the data structure when backtracking
    int* prevPartialAssignment = NULL;
    int* prevNoOfVarPerNoGood = NULL;
    int* prevLonelyVar = NULL;
    int* noGoodStatus=NULL; //the first column of the matrix is the status of the clause

    //allocates and copies the above arrays
    storePrevState(data,&prevPartialAssignment, &prevNoOfVarPerNoGood, &prevLonelyVar, &noGoodStatus);
   
    //assigns the value to the variable
    assignValueToVar(&data, var, value);
    
    pureLiteralCheck(&data);
    
    //nothing:
    learnClause();

    //if we find a conflict we backtrack (we need to revert the state first)
    if (unitPropagation(&data) == CONFLICT) {
        revert(&data, &prevPartialAssignment, &prevNoOfVarPerNoGood, &prevLonelyVar, &noGoodStatus);
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
        revert(&data, &prevPartialAssignment, &prevNoOfVarPerNoGood, &prevLonelyVar, &noGoodStatus);
		return false;
    }
    //choose the next variable to assign
    int varToAssign = chooseVar(data.partialAssignment);

    //the check is done just for reverting purposes in case we need to backtrack
    if ((solve(data, varToAssign, TRUE) || solve(data, varToAssign, FALSE)) == false) {
        revert(&data, &prevPartialAssignment, &prevNoOfVarPerNoGood, &prevLonelyVar, &noGoodStatus);
		return false;
    }
    return true;
 }

//performs the unit propagation 
int unitPropagation(struct NoGoodData* data){
    //for each no good
    for (int i = 0; i < noNoGoods; i++) {
        //if the no good is not satisfied and it has only one variable to assign we assign it
		if (data->matrix[i][0] ==UNSATISFIED &&  data->noOfVarPerNoGood[i] == 1) {
            //lonelyVar[i] is a column index
            data->partialAssignment[data->lonelyVar[i]] = data->matrix[i][data->lonelyVar[i]] > 0 ? FALSE : TRUE;
            data->varsYetToBeAssigned--;
            //we update the status of the no good
            removeNoGoodSetsContaining(&(data->matrix), &(data->currentNoGoods),data->lonelyVar[i], (data->partialAssignment[data->lonelyVar[i]]) == TRUE ? NEGATED_LIT : POSITIVE_LIT);
            if (removeLiteralFromNoGoods(data, data->lonelyVar[i], data->partialAssignment[data->lonelyVar[i]] == TRUE ? POSITIVE_LIT : NEGATED_LIT)== CONFLICT) 
                return CONFLICT;
        }
	}
    return NO_CONFLICT;
}

//removes the literal (by assigning a value) from the no goods IF it's UNASSIGNED and shows up with only one sign (in the remaining no goods)
void pureLiteralCheck(struct NoGoodData* data){

    for (int i = 1; i < noVars+1; i++) {
        if(data->partialAssignment[i] == UNASSIGNED){
            if (varBothNegatedAndNot[i] == APPEARS_ONLY_POS) {
			    data->partialAssignment[i] = FALSE;
                data->varsYetToBeAssigned--;
                removeNoGoodSetsContaining(&(data->matrix), &(data->currentNoGoods),i,POSITIVE_LIT);
		    } else if (varBothNegatedAndNot[i] == APPEARS_ONLY_NEG) {
                data->partialAssignment[i] = TRUE;
                data->varsYetToBeAssigned--;
                removeNoGoodSetsContaining(&(data->matrix), &(data->currentNoGoods),i,NEGATED_LIT);
            }
        }
    }
}

//removes (assigns 'falsified' satisfied) the no goods if they contain the literal varIndex with the indicated sign
void removeNoGoodSetsContaining(int*** matrix,int *currentNoGoods,int varIndex,int sign) {

    //scan column (varIndex) of matrix
    for (int i = 0; i < noNoGoods; i++) {
		if ((*matrix)[i][varIndex] == sign && (*matrix)[i][0] != SATISFIED) {
			//remove the nogood set
            (*matrix)[i][0] = SATISFIED;
			(*currentNoGoods)--;
		}
	}
}

//returns the index of the first unassigned variable (more policies to be implemented)
int chooseVar(int * partialAssignment) {
    //return the fist unassigned var
    for (int i = 1; i < noVars + 1; i++) {
		if (partialAssignment[i] == UNASSIGNED) {
			return i;
		}
	}
    //if all vars are assigned return -1 (never)
    return -1;
}
void assignValueToVar(struct NoGoodData* data, int varToAssign, int value) {
	data->partialAssignment[varToAssign] = value;
    data->varsYetToBeAssigned--;
    removeNoGoodSetsContaining(&(data->matrix), &(data->currentNoGoods),varToAssign, value == TRUE ? NEGATED_LIT : POSITIVE_LIT);
}
void learnClause(){
    return;
    //TODO
}
//removes the literal varIndex from the nogood if the sign is the one indicated
int removeLiteralFromNoGoods(struct NoGoodData* data, int varIndex, int sign) {
	//scan column (varIndex) of matrix
	for (int i = 0; i < noNoGoods; i++) {
		if (data->matrix[i][varIndex] == sign) {
			
			//data->matrix[i][varIndex] = 0; //not necessary WE NEVER MODIFY MATRIX (except for the first col)
            
            //remove the literal
            data->noOfVarPerNoGood[i]--;
            if(data->noOfVarPerNoGood==1){
                //search and assing the literal to the lonelyVar
                for (int j = 1; j < noVars + 1; j++) {
					if (data->matrix[i][j] != NO_LIT && data->partialAssignment[j]==UNASSIGNED) {
                        data->lonelyVar[i] = j;
					}
				}
			}
			if (data->noOfVarPerNoGood[i] == 0) {
				return CONFLICT;
			}
		}
	}
	return NO_CONFLICT;
}
//performs a copy of the arrays passed
void storePrevState(struct NoGoodData data, int** prevPartialAssignment, int** prevNoOfVarPerNoGood, int** prevLonelyVar, int** noGoodStatus) {
    (*prevPartialAssignment)=(int*)calloc(noVars + 1, sizeof(int));
    (*prevNoOfVarPerNoGood) = (int*)calloc(noNoGoods, sizeof(int));
    (*prevLonelyVar) = (int*)calloc(noNoGoods, sizeof(int));
    (*noGoodStatus) = (int*)calloc(noNoGoods, sizeof(int));
    for (int i = 0; i < noVars + 1; i++) {
		(*prevPartialAssignment)[i] = data.partialAssignment[i];
	}
    for (int i = 0; i < noNoGoods; i++) {
        (*prevNoOfVarPerNoGood)[i] = data.noOfVarPerNoGood[i];
        (*prevLonelyVar)[i] = data.lonelyVar[i];
        (*noGoodStatus)[i] = data.matrix[i][0];
    }
}
//performs a copy of the arrays passed (to revert to the previous state) then it deallocates the memory
void revert(struct NoGoodData* data, int** prevPartialAssignment, int** prevNoOfVarPerNoGood, int** prevLonelyVar, int** noGoodStatus) {
    for (int i = 0; i < noVars + 1; i++) {
    	data->partialAssignment[i] = (*prevPartialAssignment)[i];
    }
    for (int i = 0; i < noNoGoods; i++) {
		data->noOfVarPerNoGood[i] = (*prevNoOfVarPerNoGood)[i];
		data->lonelyVar[i] = (*prevLonelyVar)[i];
		data->matrix[i][0] = (*noGoodStatus)[i];
	}
    free(*prevPartialAssignment);
    free(*prevNoOfVarPerNoGood);
    free(*prevLonelyVar);
    free(*noGoodStatus);
}