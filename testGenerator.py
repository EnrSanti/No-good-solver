#!/usr/bin/python3


#WARNING: THIS PROGRAM ISN't OPTIMIZED AD ALL

import random
from random import choice

filesToCreate=10

#how many clauses we want in an instance (max and min)
minNoGoods=2000
maxNoGoods=3000

minNoVars=700
maxNoVars=800


#we set the seed so we always generate that instances
random.seed(12)

osType="linux"; # "windows" or "linux" #used just to specify the directory format

#the following function is taken from: https://www.w3resource.com/python-exercises/list/python-data-type-list-exercise-145.php
def generate_random(from_,to_, alreadyInNoGood):
    result = choice([i for i in range(from_,to_) if i not in alreadyInNoGood])
    return result

def generateNoGood(novars,actualVars):
	nogoodVars=random.randint(1, int( novars/1.5))
	alreadyInNoGood=[]
	for varInRow in range(0,nogoodVars):	
		#we want to have a variable or its negation
		var=generate_random(1, novars,alreadyInNoGood)*random.choice([-1, 1])
		f.write(str(var)+' ')
		fSAT.write(str(-var)+' ')
		alreadyInNoGood.append(var)

		if(var in actualVars):
			actualVars.remove(var)
		if(-var in actualVars):
			actualVars.remove(-var)
	
	f.write('0\n')
	fSAT.write('0\n')


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
	actualVars=[]
	print("clauses:"+str(nogood))
	with open(directoryPathNG+'test_'+str(i)+'.txt', 'w') as f:
		with open(directoryPathSAT+'test_'+str(i)+'.txt', 'w') as fSAT:
			f.write('c test automatically generated\nc\nc\n')
			fSAT.write('c test automatically generated\nc\nc\n')
			f.write('p nogood '+str(novars)+' '+str(nogood)+'\n')
			fSAT.write('p cnf '+str(novars)+' '+str(nogood)+'\n')
			actualVars=list(range(1,novars+1, 1))
			#we define a nogood / clause
			for row in range(0,nogood-1):
				generateNoGood(novars,actualVars)

			#we now write the last clause with all the vars not showing up
			#this has been done since minisat can spot and eliminate "non existing vars" while my sat solver not, leading to non equal comparison
			#this is also assumable since in general a human wouldn't declare more variables than what there actually are
			if(len(actualVars)==0):
				#we simply generate a random clause again:
				generateNoGood(novars,actualVars)
			else:
				for i in actualVars:
					f.write(str(i)+' ')
					fSAT.write('-'+str(i)+' ')
				f.write('0\n')
				fSAT.write('0\n')
			f.close()
			fSAT.close()

