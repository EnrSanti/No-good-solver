#!/usr/bin/python3

import random
from random import choice

filesToCreate=30

#how many clauses we want in an instance (max and min)
minNoGoods=100
maxNoGoods=200

minNoVars=50
maxNoVars=150


#we set the seed so we always generate that instances
random.seed(12)

osType="linux"; # "windows" or "linux" #used just to specify the directory format

#the following function is taken from: https://www.w3resource.com/python-exercises/list/python-data-type-list-exercise-145.php
def generate_random(from_,to_, alreadyInNoGood):
    result = choice([i for i in range(from_,to_) if i not in alreadyInNoGood])
    return result


directoryPathNG=""
directoryPathSAT=""

if(osType=="windows"):
	directoryPathNG="testsNG\\"
	directoryPathSAT="testsSAT\\"
else:
	directoryPathNG="testsNG/"
	directoryPathSAT="testsSAT/"


#we generate the different .txt files
for i in range(1,filesToCreate+1):
	#the size of n
	nogood=random.randint(minNoGoods, maxNoGoods)
	novars=random.randint(minNoVars, maxNoVars)
	print("clauses:"+str(nogood))
	with open(directoryPathNG+'test_'+str(i)+'.txt', 'w') as f:
		with open(directoryPathSAT+'test_'+str(i)+'.txt', 'w') as fSAT:
			f.write('c test automatically generated\nc\nc\n')
			fSAT.write('c test automatically generated\nc\nc\n')
			f.write('p nogood '+str(novars)+' '+str(nogood)+'\n')
			fSAT.write('p cnf '+str(novars)+' '+str(nogood)+'\n')
			#we define a nogood / clause
			for row in range(0,nogood):
				nogoodVars=random.randint(1, int( novars/2))
				alreadyInNoGood=[]
				for varInRow in range(0,nogoodVars):	
					#we want to have a variable or its negation
					var=generate_random(1, novars,alreadyInNoGood)*random.choice([-1, 1])
					f.write(str(var)+' ')
					fSAT.write(str(-var)+' ')
					alreadyInNoGood.append(var)
				f.write('0\n')
				fSAT.write('0\n')
			f.close()
			fSAT.close()

