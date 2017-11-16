from pulp import *
from random import randint
import numpy as np
import pulp
from itertools import product
n=3
#weight of distribution p
wp = [randint(1,50) for i in range(n)]
#weights of distribution q
wq = [randint(1,50) for i in range(n)]
sumwp = sum(wp)
sumwq = sum(wq)
minsum = min(sumwq,sumwp)
d = [[randint(1,50) for j in range(n)] for i in range(n)]
prob=LpProblem("EMD",LpMinimize) 
ingredients = [['f%d%d'%(j,i) for i in range(n)]for j in range(n)]
print ingredients
temp = sum(ingredients,[])
temp1 = sum(d,[])
print temp1
x = pulp.LpVariable.dict(
    '%s', temp, lowBound=0,cat=pulp.LpInteger)
scores = dict(zip(temp,temp1))
prob += sum([scores[i] * x[i] for i in temp])
prob += sum([x[i] for i in temp]) == minsum
for i in range(n):
	prob += sum(x[temp[n*i+j]] for j in range(n)) <= wp[i]
for j in range(n):
	prob += sum(x[temp[n*i+j]] for i in range(n)) <= wq[j]
print prob
solver = pulp.solvers.PULP_CBC_CMD()
prob.solve(solver)
for v in prob.variables():
	print v.name,"=",v.varValue
print "objective=", value(prob.objective) 

