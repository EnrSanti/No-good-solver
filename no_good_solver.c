#include <stdio.h>
#include <string.h>
#include <stdbool.h> 

void readFile(const char *);
void printError(char *);
void popualteMatrix(FILE*);

int noVars=-1; //the number of vars
int noNoGoods; //the no of clauses
int *matrix;
void main(int argc, char const *argv[]){
	
	if(argc!=2){
		printError("ERROR: Insert the file path");
		return;
	}
	readFile(argv[1]);


	free(matrix)
}



//reads the content of a simil DMACS file (not the fanciest function but it's called just once)
void readFile(const char *str){

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

    fscanf (ptr, "%d", &noVars);      
    fscanf (ptr, "%d", &noNoGoods);

	printf("number of vars: %d \n",noVars);
    printf("number of nogoods: %d \n",noNoGoods);

    popualteMatrix(ptr);
    
    fclose(ptr);
}

void popualteMatrix(FILE* ptr){

	int *matrix = (int *)calloc(noNoGoods * noVars * sizeof(int));
	int clauseCounter=noNoGoods;
	int literal=0;

					
	int offset = i * noVars + j;
	while(clauseCounter>0){
    	literal = fscanf (file, "%d", &i);
    	if(literal!=0)
    		matrix[offset]=literal;

    }

	
}
//prints str with "ERROR" in front of it
void printError(char * str){
	printf("ERROR: %s \n",str);
}