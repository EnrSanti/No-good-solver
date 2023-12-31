#include <stdio.h>
#include <string.h>
#include <stdbool.h> 
#include <stdlib.h>
#include "common.h"

bool solve(struct NoGoodData, int, int);
int unitPropagation(struct NoGoodData*);
void pureLiteralCheck(struct NoGoodData*);
void removeNoGoodSetsContaining(int***, int*, int**,int *, int, int);
int chooseVar(int*, int*);
void assignValueToVar(struct NoGoodData*, int, int);
int removeLiteralFromNoGoods(struct NoGoodData*, int, int);

void popualteMatrix(FILE*, struct NoGoodData*);
int readFile_allocateMatrix(const char*, struct NoGoodData*);
void allocateMatrix(int***);
void deallocateMatrix(int***, int **, int **,int **,int**);

void storePrevState(struct NoGoodData, int**, int**, int**, int**, int**);
void revert(struct NoGoodData*, int**, int**, int**, int**, int**);

//void backJump();
void printMatrix(int**);
void printVarArray(int*);
void learnClause();
void printError(char*);

int noVars = 0; //the number of vars in the model
int noNoGoods = 0; //the no of clauses (initially) in the model
//int *varBothNegatedAndNot = NULL; //a int array that holds the status of the variables in the clauses (see the defines above)
bool breakSearchAfterOne = true; //if true, the search will stop after the first solution is found
bool solutionFound = false; //if true, a solution was found, used to stop the search


void main(int argc, char const* argv[]) {

    //if the user didn't insert the file path or typed more
    if (argc != 2) {
        printError("Insert the file path");
        return;
    }
    //create the strucure
    struct NoGoodData data;
    //printf("%s", string);
    //we populate it with the data from the file
    if(readFile_allocateMatrix(argv[1], &data)==-1){
    	return;
    }
    //print the matrix
    //printMatrix(data.matrix);
    printf("\n");


    pureLiteralCheck(&data);
    if (unitPropagation(&data) == CONFLICT) {
      
        printf("\n\n\n**********UNSATISFIABLE**********\n\n\n");
        deallocateMatrix(&(data.matrix), &(data.partialAssignment), &(data.noOfVarPerNoGood), &(data.lonelyVar), &(data.varsAppearingInRemainingNoGoodsPositiveNegative));
        return;
    }


    //if we somehow already have an assignment, we can skip the search
    if (data.currentNoGoods == 0) {
        printf("\n\n\n**********SATISFIABLE**********\n\n\n");
        printf("Assignment:\n");
        printVarArray(data.partialAssignment);
        deallocateMatrix(&(data.matrix),&(data.partialAssignment),&(data.noOfVarPerNoGood),&(data.lonelyVar), &(data.varsAppearingInRemainingNoGoodsPositiveNegative));
        return;
    }


    //we choose a variable and we start the search
    int varToAssign = chooseVar(data.partialAssignment, data.varsAppearingInRemainingNoGoodsPositiveNegative);

    if (solve(data, varToAssign, TRUE) || solve(data, varToAssign, FALSE)) {
        printf("\n\n**********SATISFIABLE**********\n\n\n");
    } else {
        printf("\n\n**********UNSATISFIABLE**********\n\n\n"); 
    }
    deallocateMatrix(&(data.matrix),&(data.partialAssignment),&(data.noOfVarPerNoGood),&(data.lonelyVar), &(data.varsAppearingInRemainingNoGoodsPositiveNegative));
    
}



//reads the content of a simil DMACS file and populates the data structure
// (not the fanciest function but it's called just once)
int readFile_allocateMatrix(const char* str, struct NoGoodData* data) {

    FILE* ptr;
    char ch;
    ptr = fopen(str, "r");

    if (NULL == ptr) {
        printError("No such file or can't be opened");
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
    

    char pNG[9];
    fgets(pNG, 8, ptr);
    if(strcmp(pNG," nogood")!=0){
    	printf("File format is not correct\n");
    	return -1;
    }

    //ignore return value for now
    fscanf(ptr, "%d", &noVars);
    fscanf(ptr, "%d", &noNoGoods);

    printf("number of vars: %d \n", noVars);
    printf("number of nogoods: %d \n", noNoGoods);

    data->currentNoGoods = noNoGoods;
    data->varsYetToBeAssigned = noVars;

    popualteMatrix(ptr, data);

    fclose(ptr);
    return 0;
}

//subprocedure called by readFile_allocateMatrix it populates the data structure and other arrays such as varBothNegatedAndNot
void popualteMatrix(FILE* ptr, struct NoGoodData* data) {

    allocateMatrix(&(data->matrix));
    //varBothNegatedAndNot = (int *)calloc(noVars + 1, sizeof(int));
    data->noOfVarPerNoGood = (int*)calloc(noNoGoods, sizeof(int));
    data->lonelyVar = (int*)calloc(noNoGoods, sizeof(int));
    //allocate the partial assignment array
    data->partialAssignment = (int*)calloc(noVars + 1, sizeof(int));
    //data->varsAppearingInRemainingNoGoods=(int *)calloc(noVars + 1, sizeof(int));
    data->varsAppearingInRemainingNoGoodsPositiveNegative = (int*)calloc(2 * (noVars + 1), sizeof(int));

    /*for(int i = 0; i < noVars + 1; i++) {
        varBothNegatedAndNot[i] = FIRST_APPEARENCE;
    }*/

    int clauseCounter = 0;
    int literal = 0;
    while (!feof(ptr) && clauseCounter < noNoGoods) {

        //no idea why fscanf READS positive number as negative and vv (on Windows) 
        fscanf(ptr, "%d", &literal);
        if (literal == 0) {
            data->matrix[clauseCounter][0] = UNSATISFIED; //the first cell of the matrix is the status of the clause
            clauseCounter++;
        }
        else {

            int sign = literal > 0 ? POSITIVE_LIT : NEGATED_LIT;
            data->matrix[clauseCounter][literal * sign] = sign;
            data->noOfVarPerNoGood[clauseCounter]++;
            //if i have more vars i won't read this, so it can contain a wrong value (if the literal is just one the value will be correct)
            data->lonelyVar[clauseCounter] = literal * sign;
            //prima ho i negativi poi i positivi, per semplicità di accesso
            data->varsAppearingInRemainingNoGoodsPositiveNegative[(noVars + 1) * ((int)(1 + sign) / 2) + literal * sign]++;

            /*
            //populate the varBothNegatedAndNot array
            if(varBothNegatedAndNot[literal * sign]==FIRST_APPEARENCE)
                varBothNegatedAndNot[literal * sign]=sign;
            if(varBothNegatedAndNot[literal * sign]==APPEARS_ONLY_POS && sign==NEGATED_LIT)
                varBothNegatedAndNot[literal * sign]=APPEARS_BOTH;
            if(varBothNegatedAndNot[literal * sign]==APPEARS_ONLY_NEG && sign==POSITIVE_LIT)
                varBothNegatedAndNot[literal * sign]=APPEARS_BOTH;
                */
        }
    }
    //we assign to true possible missing variables
    for (int i = 1; i < noVars + 1; i++) {
        if (data->varsAppearingInRemainingNoGoodsPositiveNegative[i] + data->varsAppearingInRemainingNoGoodsPositiveNegative[i + (noVars + 1)] == 0) {
            data->partialAssignment[i] = TRUE;
            data->varsYetToBeAssigned--;
        }
    }

}

//prints str with "ERROR" in front of it
void printError(char* str) {
    printf("ERROR: %s \n", str);
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

//allocates the matrix
void allocateMatrix(int*** matrix) {
    *matrix = (int**)calloc(noNoGoods, sizeof(int*));
    for (int i = 0; i < noNoGoods; i++) {
        (*matrix)[i] = (int*)calloc(noVars + 1, sizeof(int));
    }

}

//deallocates the matrix and the other data
void deallocateMatrix(int*** matrix, int **partialAssignment, int **noOfVarPerNoGood, int **lonelyVar, int **varsAppearingInRemainingNoGoodsPositiveNegative) {

    for (int i = 0; i < noNoGoods; i++) {
        free((*matrix)[i]);
    }
    free((*matrix));
    free(*partialAssignment);
    free(*noOfVarPerNoGood);
    free(*lonelyVar);
    free(*varsAppearingInRemainingNoGoodsPositiveNegative);

}

//prints the content of the array passed (which for simplicity is an array of len noVars+1)
//the method is not general on purpose
void printVarArray(int* array) {
    for (int i = 1; i < noVars + 1; i++) {
        printf("%d  ", array[i]);
    }
}

//the main method responsible for calling unitpropagation, pureLiteralCheck and assignValueToVar
//performs the search by recursively calling itself twice (once for each value of the variable)
bool solve(struct NoGoodData data, int var, int value) {

    //printf("currentLonelyrent no goods: %d, current vars yet: %d assign var: %d=%d\n", data.currentNoGoods, data.varsYetToBeAssigned, var, value);

    //if we want to stop after the first solution and it's already found
    if (solutionFound && breakSearchAfterOne)
        return true;

    

    //local variables which will be used to revert the state of the data structure when backtracking
    int* prevPartialAssignment = NULL;
    int* prevNoOfVarPerNoGood = NULL;
    int* prevLonelyVar = NULL;
    int* prevVarsAppearingInRemainingNoGoods = NULL;
    int* noGoodStatus = NULL; //the first column of the matrix is the status of the clause

    //allocates and copies the above arrays
    storePrevState(data, &prevPartialAssignment, &prevNoOfVarPerNoGood, &prevLonelyVar, &noGoodStatus, &prevVarsAppearingInRemainingNoGoods);

    //assigns the value to the variable
    assignValueToVar(&data, var, value);

    pureLiteralCheck(&data);


    //nothing:
    learnClause();
    //if we find a conflict we backtrack (we need to revert the state first)
    if (unitPropagation(&data) == CONFLICT) {
        //printf("BACKTRACK\n");
        revert(&data, &prevPartialAssignment, &prevNoOfVarPerNoGood, &prevLonelyVar, &noGoodStatus, &prevVarsAppearingInRemainingNoGoods);
        return false;
    }
    //if we want to stop after the first solution and it's already found

    //if the partialAssignment satisfies (falsifies) all the clauses we have found a solution
    if (data.currentNoGoods == 0) {
        printf("Assignment:\n");
        printVarArray(data.partialAssignment);
        solutionFound = true;
        return true;
    }
    //if there are no more variables to assign (AND having previously checked that not all the no good are sat) we backtrack
    if (data.varsYetToBeAssigned == 0) {
        revert(&data, &prevPartialAssignment, &prevNoOfVarPerNoGood, &prevLonelyVar, &noGoodStatus, &prevVarsAppearingInRemainingNoGoods);
        return false;
    }
    //choose the next variable to assign
    int varToAssign = chooseVar(data.partialAssignment, data.varsAppearingInRemainingNoGoodsPositiveNegative);

    //the check is done just for reverting purposes in case we need to backtrack
    if ((solve(data, varToAssign, TRUE) || solve(data, varToAssign, FALSE)) == false) {
        revert(&data, &prevPartialAssignment, &prevNoOfVarPerNoGood, &prevLonelyVar, &noGoodStatus, &prevVarsAppearingInRemainingNoGoods);
        return false;
    }
    return true;
}

//performs the unit propagation (quadratic wrt no goods in worst case, can be improved obviously)
int unitPropagation(struct NoGoodData* data) {
    //for each no good
    for (int i = 0; i < noNoGoods; i++) {
        //if the no good is not satisfied and it has only one variable to assign we assign it
        if (data->matrix[i][0] == UNSATISFIED && data->noOfVarPerNoGood[i] == 1) {
            //lonelyVar[i] is a column index
            if (data->partialAssignment[data->lonelyVar[i]] != UNASSIGNED && data->partialAssignment[data->lonelyVar[i]] == data->matrix[i][data->lonelyVar[i]])
                return CONFLICT;
            data->partialAssignment[data->lonelyVar[i]] = data->matrix[i][data->lonelyVar[i]] > 0 ? FALSE : TRUE;
            data->varsYetToBeAssigned--;
            //we update the status of the no good
            removeNoGoodSetsContaining(&(data->matrix), &(data->currentNoGoods), &(data->varsAppearingInRemainingNoGoodsPositiveNegative), data->partialAssignment,data->lonelyVar[i], (data->partialAssignment[data->lonelyVar[i]]) == TRUE ? NEGATED_LIT : POSITIVE_LIT);
            if (removeLiteralFromNoGoods(data, data->lonelyVar[i], data->partialAssignment[data->lonelyVar[i]] == TRUE ? POSITIVE_LIT : NEGATED_LIT) == CONFLICT)
                return CONFLICT;
        }
    }
    return NO_CONFLICT;
}

//removes the literal (by assigning a value) from the no goods IF it's UNASSIGNED and shows up with only one sign (in the remaining no goods)
void pureLiteralCheck(struct NoGoodData* data) {

    for (int i = 1; i < noVars + 1; i++) {
        if (data->partialAssignment[i] == UNASSIGNED) {
            //if it appears only positively then we set it to false
            if (data->varsAppearingInRemainingNoGoodsPositiveNegative[i] == 0) {
                data->partialAssignment[i] = FALSE;
                data->varsYetToBeAssigned--;
                removeNoGoodSetsContaining(&(data->matrix), &(data->currentNoGoods), &(data->varsAppearingInRemainingNoGoodsPositiveNegative), data->partialAssignment,i, POSITIVE_LIT);
                //if appears only negatively we set it true
            }
            else if (data->varsAppearingInRemainingNoGoodsPositiveNegative[i + (noVars + 1)] == 0) {
                data->partialAssignment[i] = TRUE;
                data->varsYetToBeAssigned--;
                removeNoGoodSetsContaining(&(data->matrix), &(data->currentNoGoods), &(data->varsAppearingInRemainingNoGoodsPositiveNegative),data->partialAssignment, i, NEGATED_LIT);
            }
        }
    }
}

//removes (assigns 'falsified' satisfied) the no goods if they contain the literal varIndex with the indicated sign
void removeNoGoodSetsContaining(int*** matrix, int* currentNoGoods, int** varsAppearingInRemainingNoGoodsPositiveNegative,int* partialAssignment, int varIndex, int sign) {

    //scan column (varIndex) of matrix
    for (int i = 0; i < noNoGoods; i++) {
        if ((*matrix)[i][varIndex] == sign && (*matrix)[i][0] != SATISFIED) {
            //remove the nogood set
            (*matrix)[i][0] = SATISFIED;
            (*currentNoGoods)--;
            //we decrease the vars that were in the NG
            for (int j = 1; j < noVars + 1; j++) {
                if ((*matrix)[i][j] != 0 && (partialAssignment[j]==UNASSIGNED )&&(*varsAppearingInRemainingNoGoodsPositiveNegative)[j + (noVars + 1) * ((int)(1 + (*matrix)[i][j]) / 2)] > 0)
                    (*varsAppearingInRemainingNoGoodsPositiveNegative)[j + (noVars + 1) * ((int)(1 + (*matrix)[i][j]) / 2)]--;
            }
        }
    }
    //we set to 0 both the negated version and the positive one of the vars assigned (i.e this method is called when an assignemnt is made)
    (*varsAppearingInRemainingNoGoodsPositiveNegative)[varIndex] = 0;
    (*varsAppearingInRemainingNoGoodsPositiveNegative)[varIndex + noVars + 1] = 0;
}

//returns the index of the first unassigned variable (more policies to be implemented)
int chooseVar(int* partialAssignment, int* varsAppearingInRemainingNoGoodsPositiveNegative) {
    //return the fist unassigned var
    for (int i = 1; i < noVars + 1; i++) {
        if (partialAssignment[i] == UNASSIGNED && varsAppearingInRemainingNoGoodsPositiveNegative[i] + varsAppearingInRemainingNoGoodsPositiveNegative[i + noVars + 1] > 0) {
            return i;
        }
    }
    //printf("NEVERE HEREEEEEEEEEEEEEEEEEEEEEEEEEEEe");
    //if all vars are assigned return -1 (never)
    return -1;
}
void assignValueToVar(struct NoGoodData* data, int varToAssign, int value) {
    if (data->partialAssignment[varToAssign] != UNASSIGNED)
        return;
    data->partialAssignment[varToAssign] = value;
    data->varsYetToBeAssigned--;
    removeNoGoodSetsContaining(&(data->matrix), &(data->currentNoGoods), &(data->varsAppearingInRemainingNoGoodsPositiveNegative),data->partialAssignment, varToAssign, value == TRUE ? NEGATED_LIT : POSITIVE_LIT);
    removeLiteralFromNoGoods(data, varToAssign, value == TRUE ? POSITIVE_LIT : NEGATED_LIT);
}
void learnClause() {
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
            if (data->noOfVarPerNoGood[i] == 1) {
                //search and assing the literal to the lonelyVar
                for (int j = 1; j < noVars + 1; j++) {
                    if (data->matrix[i][j] != NO_LIT && data->partialAssignment[j] == UNASSIGNED) {
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
void storePrevState(struct NoGoodData data, int** prevPartialAssignment, int** prevNoOfVarPerNoGood, int** prevLonelyVar, int** noGoodStatus, int** prevVarsAppearingInRemainingNoGoodsPositiveNegative) {
    (*prevPartialAssignment) = (int*)calloc(noVars + 1, sizeof(int));
    (*prevVarsAppearingInRemainingNoGoodsPositiveNegative) = (int*)calloc(2 * (noVars + 1), sizeof(int));
    (*prevNoOfVarPerNoGood) = (int*)calloc(noNoGoods, sizeof(int));
    (*prevLonelyVar) = (int*)calloc(noNoGoods, sizeof(int));
    (*noGoodStatus) = (int*)calloc(noNoGoods, sizeof(int));
    for (int i = 0; i < noVars + 1; i++) {
        (*prevPartialAssignment)[i] = data.partialAssignment[i];
        (*prevVarsAppearingInRemainingNoGoodsPositiveNegative)[i] = data.varsAppearingInRemainingNoGoodsPositiveNegative[i];
        (*prevVarsAppearingInRemainingNoGoodsPositiveNegative)[i + (noVars + 1)] = data.varsAppearingInRemainingNoGoodsPositiveNegative[i + (noVars + 1)];
    }
    for (int i = 0; i < noNoGoods; i++) {
        (*prevNoOfVarPerNoGood)[i] = data.noOfVarPerNoGood[i];
        (*prevLonelyVar)[i] = data.lonelyVar[i];
        (*noGoodStatus)[i] = data.matrix[i][0];
    }
}
//performs a copy of the arrays passed (to revert to the previous state) then it deallocates the memory
void revert(struct NoGoodData* data, int** prevPartialAssignment, int** prevNoOfVarPerNoGood, int** prevLonelyVar, int** noGoodStatus, int** prevVarsAppearingInRemainingNoGoodsPositiveNegative) {
    for (int i = 0; i < noVars + 1; i++) {
        data->partialAssignment[i] = (*prevPartialAssignment)[i];
        data->varsAppearingInRemainingNoGoodsPositiveNegative[i] = (*prevVarsAppearingInRemainingNoGoodsPositiveNegative)[i];
        data->varsAppearingInRemainingNoGoodsPositiveNegative[i + (noVars + 1)] = (*prevVarsAppearingInRemainingNoGoodsPositiveNegative)[i + (noVars + 1)];
    }
    for (int i = 0; i < noNoGoods; i++) {
        data->noOfVarPerNoGood[i] = (*prevNoOfVarPerNoGood)[i];
        data->lonelyVar[i] = (*prevLonelyVar)[i];
        data->matrix[i][0] = (*noGoodStatus)[i];
    }
    free(*prevPartialAssignment);
    free(*prevNoOfVarPerNoGood);
    free(*prevLonelyVar);
    free(*prevVarsAppearingInRemainingNoGoodsPositiveNegative);
    free(*noGoodStatus);
}
