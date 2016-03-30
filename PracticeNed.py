from __future__ import division
import pandas
import pyomo.opt
import pyomo.environ as pe
import scipy
import itertools
import logging

model = pe.ConcreteModel()
model.x = pe.Var([1,2], domain=pe.NonNegativeReals)
model.OBJ = pe.Objective(expr = 2*model.x[1] + 3*model.x[2])
model.Constraint1 = pe.Constraint(expr = 3*model.x[1] + 4*model.x[2] >= 1)
solver = pyomo.opt.SolverFactory('cplex')




