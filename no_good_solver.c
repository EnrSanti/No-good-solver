#include <stdio.h>
#include <string.h>
#include <stdbool.h> 
#include <stdlib.h>

//for the clauses
#define UNSATISFIED -2
#define SATISFIED 2

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

void readFile_allocateMatrix(const char *);
void printError(char *);
void popualteMatrix(FILE*);
void printMatrix();
void printVarArray(int *);
void allocateMatrix();
void deallocateMatrix();
void solve();
void unitPropagation();
void backJump();
void pureLiteralCheck();


int noVars=0; //the number of vars
int noNoGoods=0; //the no of clauses
int falsifiedNoGoods=0; //the number of non satisfied clauses
int **matrix=NULL; //the matrix that holds the clauses
int *partialAssignment;//we skip the cell 0 in order to maintain coherence with the var numbering
int* varBothNegatedAndNot = NULL;

void main(int argc, char const *argv[]){
	
	if(argc!=2){
		printError("Insert the file path");
		return;
	}

    
    readFile_allocateMatrix(argv[1]);
    printMatrix();
    printf("The status of the variables in the clauses: (%d doesn't appear, %d just positive, %d just negative, %d both)\n",FIRST_APPEARENCE, APPEARS_ONLY_POS,   APPEARS_ONLY_NEG,   APPEARS_BOTH);
    printVarArray(varBothNegatedAndNot);

	partialAssignment=(int *) calloc(noVars+1, sizeof(int));
	
    
	pureLiteralCheck();

    solve();

    deallocateMatrix();
}



//reads the content of a simil DMACS file (not the fanciest function but it's called just once)
void readFile_allocateMatrix(const char *str){

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

    falsifiedNoGoods=noNoGoods;

    popualteMatrix(ptr, varBothNegatedAndNot);
    
    fclose(ptr);
}

void popualteMatrix(FILE* ptr){

   
	allocateMatrix();
    varBothNegatedAndNot = (int *)calloc(noVars + 1, sizeof(int));
    for(int i = 0; i < noVars + 1; i++) {
		varBothNegatedAndNot[i] = FIRST_APPEARENCE;
	}
    int clauseCounter=0;
	int literal=0;

    while(!feof(ptr) && clauseCounter<noNoGoods){
		fscanf (ptr, "%d", &literal);
		if(literal==0){
			matrix[clauseCounter][0]=UNSATISFIED; 
			clauseCounter++;
		}else{
            int sign = literal > 0 ? POSITIVE_LIT : NEGATED_LIT;
            matrix[clauseCounter][literal*sign] = sign;

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
void printMatrix(){
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
void allocateMatrix(){
	matrix = (int **) calloc(noNoGoods, sizeof(int *));
	for (int i = 0; i < noNoGoods; i++){
 		matrix[i] = (int *) calloc(noVars+1, sizeof(int));
	}
}
void deallocateMatrix(){

	for (int i = 0; i < noNoGoods; i++){
 		free(matrix[i]);
	}
	free(matrix);
}
void printVarArray(int *array) {
    for (int i =1; i < noVars+1; i++) {
        printf("%d  ", array[i]);
    }
}

void solve(){
    if (falsifiedNoGoods==0) {
        printf("SATISFIABLR\n");
		return;
    }
}

void unitPropagation(){
    
}
void pureLiteralCheck(){
    for (int i = 1; i < noVars; i++) {
        if (varBothNegatedAndNot[i] == APPEARS_ONLY_POS) {
			partialAssignment[i] = FALSE;
            falsifiedNoGoods=-noclauseswuth;
		} else if (varBothNegatedAndNot[i] == APPEARS_ONLY_NEG) {
			partialAssignment[i] = TRUE;
            falsifiedNoGoods = -noclauseswuth;
        }
    }
}

void backJump(){

}
