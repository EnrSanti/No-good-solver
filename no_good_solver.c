#include <stdio.h>
#include <string.h>
#include <stdbool.h> 
#include <stdlib.h>

#define UNSATISFIED -2
#define SATISFIED 2

void readFile_allocateMatrix(const char *);
void printError(char *);
void popualteMatrix(FILE*);
void printMatrix();
void allocateMatrix();
void deallocateMatrix();
void solve();
void unitPropagation();
void backJump();


int noVars=0; //the number of vars
int noNoGoods=0; //the no of clauses
int nonSatisfied=0; //the number of non satisfied clauses
int **matrix=NULL; //the matrix that holds the clauses


void main(int argc, char const *argv[]){
	
	if(argc!=2){
		printError("Insert the file path");
		return;
	}

	readFile_allocateMatrix(argv[1]);
    printMatrix();
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

    nonSatisfied=noNoGoods;

    popualteMatrix(ptr);
    
    fclose(ptr);
}

void popualteMatrix(FILE* ptr){


	allocateMatrix();
	int clauseCounter=0;
	int literal=0;

    while(!feof(ptr) && clauseCounter<noNoGoods){
		fscanf (ptr, "%d", &literal);
		if(literal==0){
			matrix[clauseCounter][0]=UNSATISFIED; 
			clauseCounter++;
		}else{
            if (literal > 0)
                matrix[clauseCounter][literal] = 1;
            else
                matrix[clauseCounter][-literal] = -1;
		}
	}

	
}
//prints str with "ERROR" in front of it
void printError(char * str){
	printf("ERROR: %s \n",str);
}
void printMatrix(){
	for (int i = 0; i < noNoGoods; i++){
		for (int j = 0; j < noVars+1; j++){
			if(matrix[i][j]<0)
				printf("%d ", matrix[i][j]);
			else
				printf(" %d ", matrix[i][j]);
		}
		printf("\n");
	}
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

void solve(){
    if (nonSatisfied==0) {
        printf("SATISFIABLR\n");

		return;
    }
}

void unitPropagation()
{
}

void backJump()
{
}
