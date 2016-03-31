import pyomo.environ as pe
import pandas

model = pe.ConcreteModel()

model.pilots = pe.Set(initialize=['p1','p2','p3','p4','p5'])
model.rank = pe.Set(initialize=['CPT','FO'])
model.fleet = pe.Set(initialize=[330,320])
model.base = pe.Set(initialize=['b1','b2'])
model.time = pe.Set(initialize=[1,2,3,4,5])

fromPos = pandas.read_csv('from_pos.csv')
fromPos.set_index(['rank','fleet','base'], inplace=True)

toPos = pandas.read_csv('to_pos.csv')
toPos.set_index(['rank','fleet','base'], inplace=True)

fixedPos = pandas.read_csv('fixed_pos.csv')
fixedPos.set_index(['rank','fleet','base'], inplace=True)

demand = pandas.read_csv('demand.csv')
demand.set_index(['rank','fleet','base','time'], inplace=True)

model.Y = pe.Var(model.pilots*model.rank*model.fleet*model.base*model.time, domain=pe.Binary)
model.S = pe.Var(model.rank*model.fleet*model.base*model.time, domain=pe.Binary)

def demand_rule(model, r, f, b, t):
    cur_position = '%s-%s-%s'%(r,f,b) 
    rhs = fixed_position[ cur_position ]
    for p in toPos.ix[ (r,f,b), 'pilot'].values:
        rhs = rhs + model.Y[p,r,f,b,t]
    for p in fromPos.ix[ (r,f,b), 'pilot'].values:
        rhs = rhs - model.Y[p,r,f,b,t]   
    rhs = rhs + model.S[r,f,b,t]
    return rhs <= demand.ix[ (r,f,b,t), 'demand']

model.Demand = pe.Constraint(model.rank*model.fleet*model.base*model.time, rule=demand_rule)
