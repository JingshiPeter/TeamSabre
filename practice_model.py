from __future__ import division
import pyomo.environ as pe

model = pe.AbstractModel()

model.m = pe.Param(within=NonNegativeIntegers)
model.n = pe.Param(within=NonNegativeIntegers)

model.I = pe.RangeSet(1, model.m)
model.J = pe.RangeSet(1, model.n)

model.a = pe.Param(model.I, model.J)
model.b = pe.Param(model.I)
model.c = pe.Param(model.J)

# the next line declares a variable indexed by the set J
model.x = pe.Var(model.J, domain=NonNegativeReals)

def obj_expression(model):
    return summation(model.c, model.x)

model.OBJ = pe.Objective(rule=obj_expression)

def ax_constraint_rule(model, i):
    # return the expression for the constraint for i
    return sum(model.a[i,j] * model.x[j] for j in model.J) >= model.b[i]

# the next line creates one constraint for each member of the set model.I
model.AxbConstraint = pe.Constraint(model.I, rule=ax_constraint_rule)


