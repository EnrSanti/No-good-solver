#!/usr/bin/python3

import random

filesToCreate=30

#how many clauses we want in an instance (max and min)
minNoGoods=10
maxNoGoods=20

minNoVars=5
maxNoVars=15


#we set the seed so we always generate that instances
random.seed(12)

#we generate the different .txt files
for i in range(1,filesToCreate+1):
	#the size of n
	nogood=random.randint(minNoGoods, maxNoGoods)
	novars=random.randint(minNoVars, maxNoVars)
	print("clauses:"+str(nogood))
	with open('testsNG\\test_'+str(i)+'.txt', 'w') as f:
		with open('testsSAT\\test_'+str(i)+'.txt', 'w') as fSAT:
			f.write('c test automatically generated\nc\nc\n')
			fSAT.write('c test automatically generated\nc\nc\n')
			f.write('p nogood '+str(novars)+' '+str(nogood)+'\n')
			fSAT.write('p cnf '+str(novars)+' '+str(nogood)+'\n')
			#we define a nogood / clause
			for row in range(0,nogood):
				nogoodVars=random.randint(1, int( novars/2))
				for varInRow in range(0,nogoodVars):	
					#we want to have a variable or its negation
					var=random.randint(1, novars)*random.choice([-1, 1])
					f.write(str(var)+' ')
					fSAT.write(str(-var)+' ')
				f.write('0\n')
				fSAT.write('0\n')
			f.close()
			fSAT.close()
