import pandas
import pyomo.opt
import pyomo.environ as pe
import scipy
import itertools
import cplex
import pandas
import logging

#DEFINE GLOBAL NAMES HERE
CREWDATA_CSV = 'CrewData.csv'
DEMANDDATA_CSV = 'DemandData.csv'
VACATIONDATA_CSV = 'VacationData.csv'

crew_df = pandas.read_csv(CREWDATA_CSV)
demand_df = pandas.read_csv(DEMANDDATA_CSV)
vacation_df = pandas.read_csv(VACATIONDATA_CSV)
# def get_base_dic():
# 	###finished
# 	# return a base dictionary
# 	# example : 
# 	# 	bases = {

# 	# 'b1' : {'900488', '900421'},
# 	# 'b2' : {'900201', '900424'}
    
#     #  }
	
# 	b1_ids = crew_df[crew_df.Current_Base == 1]['Crew_ID']
# 	b2_ids = crew_df[crew_df.Current_Base == 2]['Crew_ID']
# 	# print "number of pilots in base 1 is " + str(len(b1_ids))
# 	# print "number of pilots in base 2 is " + str(len(b2_ids))
# 	base_dic = {
# 	    "B1" : set(b1_ids),
# 	    "B2" : set(b2_ids),
# 	    }
# 	return base_dic

# def get_rank_dic():
# 	###rank
# 	# return a rank dictionary
# 	# example : 
# 	# ranks = {

# 	# 'cpt' : {'900201','900488', '900421'},
# 	# 'fo' : {'900424'}

# 	# }
# 	cpt_ids = crew_df[crew_df.Rank == "CPT"]['Crew_ID']
# 	fo_ids = crew_df[crew_df.Rank == "FO"]['Crew_ID']
# 	# print "number of first captains is " + str(len(cpt_ids))
# 	# print "number of first officers is " + str(len(fo_ids))
# 	rank_dic = {
# 	    "CPT" : set(cpt_ids),
# 	    "FO" : set(fo_ids)
# 	    }
# 	return rank_dic

# def get_fleet_dic():
# 	###finished
# 	# return a fleet dictionary
# 	# example :
# 	# fleets = {
# 	# 'a330' : {'900201', '900421'},
# 	# 'a320' : {'900488', '900424'}

# 	# }
# 	A320_ids = crew_df[crew_df.Cur_Fleet == "A320"]['Crew_ID']
# 	A330_ids = crew_df[crew_df.Cur_Fleet == "A330"]['Crew_ID']
# 	print "number of pilots for A320 is " + str(len(A320_ids))
# 	print "number of pilots for A330 is " + str(len(A330_ids))
# 	fleet_dic = {
# 	    "A320" : set(A320_ids),
# 	    "A330" : set(A330_ids),
# 	    }
# 	print fleet_dic
# 	return fleet_dic

# def get_orig_fleet_dic():
# 	###finished
# 	orig_fleet_dic = {
# 		"A320" : set(crew_df[crew_df.Bid_FleetChange.notnull()][crew_df.Cur_Fleet == "A320"]['Crew_ID']),
# 		"A330" : {}
# 	}
# 	print orig_fleet_dic
# 	return orig_fleet_dic

# def get_future_fleet_dic():
# 	###finished
# 	fleet_change_future_A320_ids = crew_df[crew_df.Bid_FleetChange == "A320"]['Crew_ID']
# 	fleet_change_future_A330_ids = crew_df[crew_df.Bid_FleetChange == "A330"]['Crew_ID']
# 	# print "number of pilots who bid fleetchange future A320 is " + str(len(fleet_change_future_A320_ids))
# 	# print "number of pilots who bid fleetchange future A330 is " + str(len(fleet_change_future_A330_ids))
# 	future_fleet_dic = {
# 	    "A320" : set(fleet_change_future_A320_ids),
# 	    "A330" : set(fleet_change_future_A330_ids),
# 	    }
# 	print future_fleet_dic
# 	return future_fleet_dic

# def get_orig_rank_dic():
# 	###finished
# 	orig_rank_dic = {
# 		"FO" : set(crew_df[crew_df.Bid_RankChange.notnull()][crew_df.Rank == "FO"]['Crew_ID']),
# 		"CPT" : {}
# 	}
# 	print orig_rank_dic
# 	return orig_rank_dic

# def get_future_rank_dic():
# 	###finished
# 	future_rank_dic = {
# 		"CPT" : set(crew_df[crew_df.Bid_RankChange.notnull()][crew_df.Rank == "FO"]['Crew_ID']),
# 		"FO" : {}
# 	}
# 	print future_rank_dic
# 	return future_rank_dic

# def get_orig_base_dic():
#     b1_o_ids = crew_df[(crew_df.Bid_BaseChange == 1)]['Crew_ID']
#     b2_o_ids = crew_df[(crew_df.Bid_BaseChange == 2)]['Crew_ID']
#     baseBidOriginal_dict = {
# 	"B1" : set(b2_o_ids),
# 	"B2" : set(b1_o_ids),
# 	}
#     return baseBidOriginal_dict
    
# def get_future_base_dic():
#     bF_ids = crew_df[(crew_df.Bid_BaseChange == 1) | (crew_df.Bid_BaseChange == 2)]['Crew_ID']
#     #print "number of pilots wanting base change is " + str(len(bO_ids))
#     baseBid_dict = {
# 	"Base Bid" : set(bF_ids),
# 	}
#     return baseBid_dict

# orig_rank_dic = get_orig_rank_dic()
# orig_fleet_dic = get_orig_fleet_dic()
# orig_base_dic = get_orig_base_dic()

# future_rank_dic = get_future_rank_dic()
# future_fleet_dic = get_future_fleet_dic()
# future_base_dic = get_future_base_dic()

# def get_orig_position(base, fleet, rank):
# 	# return non-fix group pilot ids whose orig position is input
# 	return set(orig_base_dic[base] and orig_fleet_dic[fleet] and orig_rank_dic[rank])

# def get_future_position(base, fleet, rank):
# 	# return non-fix group pilot ids whose future position is input
# 	return set(future_base_dic[base] and future_fleet_dic[fleet] and future_rank_dic[rank])


# base_dic = get_base_dic()
# rank_dic = get_rank_dic()
# fleet_dic = get_fleet_dic()
# def get_position(base, fleet, rank):
# 	# return fix group pilot ids whose position is input
# 	return set(base_dic[base] and fleet_dic[fleet] and rank_dic[rank])

def get_demand(rank, fleet, base, week):
	# example: base = "B1", fleet = "A330", rank = "FO", week = 0
	# return the demand at B1, A330, FO of week 0
	return demand_df['B'+ str(base) + '_' + fleet[1:] + rank][week]

def get_nonfix_pilots():
	return set(crew_df[(crew_df.Bid_BaseChange.notnull()) | (crew_df.Bid_FleetChange.notnull())| (crew_df.Bid_RankChange.notnull())]['Crew_ID'])

def get_all_pilots():
	return set(crew_df[crew_df.Rank != "SIM_INS"]['Crew_ID'])

####trainer Pilots
trainers = set(crew_df[(crew_df.Instructor.notnull())]['Crew_ID'])

####fixedPos
def print_duplicate(a):
	print [item for item, count in collections.Counter(a).items() if count > 1]

nonfixed_df = crew_df[(crew_df.Bid_BaseChange.notnull()) | (crew_df.Bid_FleetChange.notnull())| (crew_df.Bid_RankChange.notnull())]
fixed_df = crew_df[(~crew_df.Bid_BaseChange.notnull()) & (~crew_df.Bid_FleetChange.notnull()) & (~crew_df.Bid_RankChange.notnull())]

#### toPos
topos_list = []
rank_change = set(crew_df[(crew_df.Bid_RankChange.notnull())]['Crew_ID'])
fleet_change = set(crew_df[(crew_df.Bid_FleetChange.notnull())]['Crew_ID'])
base_change = set(crew_df[(crew_df.Bid_BaseChange.notnull())]['Crew_ID'])
for pilot in set(nonfixed_df['Crew_ID']):
	cur = [pilot]
	pilot_item = crew_df[crew_df.Crew_ID == pilot]
	if pilot in rank_change:
		print pilot_item
		cur.append('CPT')
		cur.append(pilot_item.Cur_Fleet.values[0])
		cur.append(pilot_item.Current_Base.values[0])
	elif pilot in fleet_change :
		cur.append(pilot_item.Rank.values[0])
		cur.append(pilot_item.Bid_FleetChange.values[0])
		cur.append(pilot_item.Current_Base.values[0])
	elif pilot in base_change :
		cur.append(pilot_item.Rank.values[0])
		cur.append(pilot_item.Cur_Fleet.values[0])
		cur.append(pilot_item.Bid_BaseChange.values[0])
	topos_list.append(cur)

toPos = pandas.DataFrame(topos_list)
toPos.columns =['ID','RANK','FLEET','BASE']

#### fromPos
frompos_list = []
for pilot in set(nonfixed_df['Crew_ID']):
	cur = [pilot]
	pilot_item = crew_df[crew_df.Crew_ID == pilot]
	cur.append(pilot_item.Rank.values[0])
	cur.append(pilot_item.Cur_Fleet.values[0])
	cur.append(pilot_item.Current_Base.values[0])
	frompos_list.append(cur)

fromPos = pandas.DataFrame(frompos_list)
fromPos.columns =['ID','RANK','FLEET','BASE']

# ALL debuged before this point

model = pe.ConcreteModel()
model.pilots = pe.Set(initialize=get_all_pilots())
variable_set=[]
from_set = []
to_set = []

for pilot in nonfixed_df['Crew_ID'].values:
        for fleet in ['A320','A330']:
            for base in [1,2]:
                for rank in ['CPT','FO']:
                	in_from = pilot in fromPos[(fromPos.RANK==rank)&(fromPos.FLEET==fleet)&(fromPos.BASE==base)]['ID'].values
                	in_to = pilot in toPos[(toPos.RANK==rank)&(toPos.FLEET==fleet)&(toPos.BASE==base)]['ID'].values
                	if(in_from or in_to):
                		variable_set.append((pilot,rank,fleet,base))
                	if in_from :
                		from_set.append((pilot,rank,fleet,base))
                	if in_to :
                		to_set.append((pilot,rank,fleet,base))

model.nonfix_pilots = pe.Set(initialize = nonfixed_df['Crew_ID'].values)
print "Number of nonfix_pilots is " + str(len(nonfixed_df['Crew_ID'].values))
model.trainer_pilots = pe.Set(initialize = trainers)
model.rank_pilots = pe.Set(initialize = rank_change)
model.fleet_pilots = pe.Set(initialize = fleet_change)
model.base_pilots = pe.Set(initialize = base_change)
model.from_pos = pe.Set(initialize = from_set)
model.to_pos = pe.Set(initialize = to_set)

model.fix_pilots = model.pilots - model.nonfix_pilots
model.rank = pe.Set(initialize=['CPT','FO'])
model.fleet = pe.Set(initialize=['A330','A320'])
model.base = pe.Set(initialize=[1,2])
model.time = pe.Set(initialize=range(26))
model.timestart = pe.Set(initialize=range(25))
model.quarterstart = pe.Set(initialize = [0,13])

model.Y = pe.Var(model.nonfix_pilots*model.rank*model.fleet*model.base*model.time, domain=pe.Binary)
model.S = pe.Var(model.rank*model.fleet*model.base*model.time, domain=pe.NonNegativeIntegers)
model.V = pe.Var(model.pilots*model.time, domain=pe.Binary)
model.T = pe.Var(model.trainer_pilots*model.time, domain=pe.Binary)
model.Trainee = pe.Var(model.fleet_pilots*model.time, domain=pe.Binary)

###shortage cost
model.short_cost = pe.Param(model.rank*model.fleet*model.base*model.time, initialize = 1000)
model.transition_cost = pe.Param(model.nonfix_pilots*model.rank*model.fleet*model.base*model.time, initialize = 10)

#demand rule
#checked
def demand_rule(model, r, f, b, t):
    rhs = len(fixed_df[(fixed_df.Rank==r)&(fixed_df.Cur_Fleet==f)&(fixed_df.Current_Base==b)]['Crew_ID'])

    for p in fixed_df[(fixed_df.Rank==r)&(fixed_df.Cur_Fleet==f)&(fixed_df.Current_Base==b)]['Crew_ID'].values:
    	# if p is in vacation
    	rhs -= model.V[p,t]

    for p in toPos[(toPos.RANK==r)&(toPos.FLEET==f)&(toPos.BASE==b)]['ID'].values:
        rhs += model.Y[p,r,f,b,t]
    #if p come to the position by a fleet change, it takes 5 weeks training
    	if(p in fleet_change and t >= 5) :
    		rhs = rhs - 1 + model.Y[p,r,f,b,t-5]
    	elif(p in fleet_change and t < 5) :
    		rhs = rhs - model.Y[p,r,f,b,t]

    for p in fromPos[(fromPos.RANK==r)&(fromPos.FLEET==f)&(fromPos.BASE==b)]['ID'].values:
        rhs = rhs - model.Y[p,r,f,b,t]

    rhs = rhs + model.S[r,f,b,t]
    return rhs >= get_demand(r,f,b,t)

model.Demand = pe.Constraint(model.rank*model.fleet*model.base*model.time, rule=demand_rule)

# model.Demand.pprint()

#at time t, a pilot should ocupy one and only one position
#checked
def pilot_pos_rule(model, pilot, t):
	lhs = pe.summation(model.Y, index = [(p, r, f, b, t) for (p, r, f, b) in variable_set if p == pilot])
	return lhs == 1
model.PositionConst = pe.Constraint(model.nonfix_pilots*model.time, rule = pilot_pos_rule)
# model.PositionConst.pprint()

# all nonfix_pilots should transit only once--"from" postion should be decreasing
def pilot_transit_rule1(model, p, r, f, b, t):
	return model.Y[p,r,f,b,t] - model.Y[p,r,f,b,t+1] >= 0
model.Transition1 = pe.Constraint(model.from_pos*model.timestart, rule = pilot_transit_rule1)
# model.Transition1.pprint()

# "to" postion should be increasing
def pilot_transit_rule2(model, p, r, f, b, t):
	return model.Y[p,r,f,b,t] - model.Y[p,r,f,b,t+1] <= 0
model.Transition2 = pe.Constraint(model.to_pos*model.timestart, rule = pilot_transit_rule2)

model.Transition2.pprint()


def get_slot(t):
	return vacation_df["Available_Vacation_Slots"][t]

 #vacation constraint. -vacation. pilot <= slot. 
def pilot_vacation_slot_exceed(model, t):
	lhs = 0
	for pilot in model.pilots :
		lhs += model.V[pilot,t]
	return lhs <= get_slot(t)  
model.pilot_vacation_slot_exceed = pe.Constraint(model.time, rule = pilot_vacation_slot_exceed)

def trainee_var_binding_rule(model, p, r, f, b, t):
	return model.Y[p,r,f,b,t] - model.Y[p,r,f,b,t+1] - model.Trainee[p, t] <= 0
model.trainee_var_binding = pe.Constraint(mode.from_pos*model.timestart, rule=trainee_var_binding_rule)

# def trainer_binding_rule(model, b, t):
# 	model.Y

### at least one vacation per quarter
def vacation_rule(model, p, t):
	lhs = 0
	for i in range(13):
		lhs += model.V[p,t+i]
	return lhs >= 0
model.Vacation = pe.Constraint(model.pilots*model.quarterstart, rule = vacation_rule)


model.total_trans_cost = pe.summation(model.transition_cost, model.Y, index = [(p, r, f, b, 25) for(p, r, f, b) in to_set ])
model.OBJ = pe.Objective(expr = pe.summation(model.short_cost, model.S) + model.total_trans_cost, sense=pe.minimize)
solver = pyomo.opt.SolverFactory('cplex')


results = solver.solve(model, tee=True, keepfiles=False)
if (results.solver.status != pyomo.opt.SolverStatus.ok):
    logging.warning('Check solver not ok?')
if (results.solver.termination_condition != pyomo.opt.TerminationCondition.optimal):  
    logging.warning('Check solver optimality?')
model.solutions.load_from(results)
#model.load(results)

print 'Daeun, Please start coding below'


print 'Mridula at time t, pilot P changed position: result visualzation'
# converting the week set into a list
list_time = list(model.time.value)
print list_time
list_nfPilots = list(model.nonfix_pilots.value)
print list_nfPilots

for pilott in list_nfPilots: 
	for i in list_time:
		for base in list(model.base.value):
			for rank in list(model.rank.value):
				for fleet in list(model.fleet.value):
					if model.Y[pilott,rank,fleet,base,i].value == 0.0 or model.Y[pilott,rank,fleet,base,i].value == 1.0:
						print str(pilott) + "__" + str(rank) + '__'+ str(fleet) + '__' + str(base) + '_week' + str(i) 
						print model.Y[pilott,rank,fleet,base,i].value 
					



# record the transition in each week



print 'Total cost = ', model.OBJ()

#instance.solutions.load_from(results)
#model.solutions.load_from(results)









