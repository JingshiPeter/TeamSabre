import pandas

#DEFINE GLOBAL NAMES HERE
CREWDATA_CSV = 'CrewData.csv'

crew_df = pandas.read_csv(CREWDATA_CSV)

def get_base_dic():
	# return a base dictionary
	# example : 
	# 	bases = {

	# 'b1' : {'900488', '900421'},
	# 'b2' : {'900201', '900424'}
	
	b1_ids = crew_df[crew_df.Current_Base == 1]['Crew_ID']
	b2_ids = crew_df[crew_df.Current_Base == 2]['Crew_ID']
	print "number of pilots in base 1 is " + str(len(b1_ids))
	print "number of pilots in base 2 is " + str(len(b2_ids))
	base_dict = {
	    "B1" : set(b1_ids),
	    "B2" : set(b2_ids),
	    }
	return base_dict

def get_rank_dic():
	# return a rank dictionary
	# example : 
	# ranks = {

	# 'cpt' : {'900201','900488', '900421'},
	# 'fo' : {'900424'}

	# }
	cpt_ids = crew_df[crew_df.Rank == "CPT"]['Crew_ID']
	fo_ids = crew_df[crew_df.Rank == "FO"]['Crew_ID']
	print "number of first captains is " + str(len(cpt_ids))
	print "number of first officers is " + str(len(fo_ids))
	rank_dic = {
	    "CPT" : set(cpt_ids),
	    "FO" : set(fo_ids)
	    }
	return rank_dic

def get_fleet_dic():
	# return a fleet dictionary
	# example :
	# fleets = {
	# 'a330' : {'900201', '900421'},
	# 'a320' : {'900488', '900424'}

	# }
	A320_ids = crew_df[crew_df.Cur_Fleet == "A320"]['Crew_ID']
	A330_ids = crew_df[crew_df.Cur_Fleet == "A330"]['Crew_ID']
	print "number of pilots for A320 is " + str(len(A320_ids))
	print "number of pilots for A330 is " + str(len(A330_ids))
	fleet_dic = {
	    "A320" : set(A320_ids),
	    "A330" : set(A330_ids),
	    }
	print fleet_dic
	return 0

def get_orig_fleet_dic():
	return 0

def get_future_fleet_dic():
	fleet_change_future_A320_ids = crew_df[crew_df.Bid_FleetChange == "A320"]['Crew_ID']
	fleet_change_future_A330_ids = crew_df[crew_df.Bid_FleetChange == "A330"]['Crew_ID']
	print "number of pilots who bid fleetchange future A320 is " + str(len(fleet_change_future_A320_ids))
	print "number of pilots who bid fleetchange future A330 is " + str(len(fleet_change_future_A330_ids))
	future_fleet_dic = {
	    "A320" : set(fleet_change_future_A320_ids),
	    "A330" : set(fleet_change_future_A330_ids),
	    }
	print future_fleet_dic
	return future_fleet_dic

def get_orig_rank_dic():
	return 0

def get_future_rank_dic():
	return 0

def get_orig_base_dic():
    b1_o_ids = crew_df[(crew_df.Bid_BaseChange == 1)]['Crew_ID']
    b2_o_ids = crew_df[(crew_df.Bid_BaseChange == 2)]['Crew_ID']
    baseBidOriginal_dict = {
	"B1" : set(b2_o_ids),
	"B2" : set(b1_o_ids),
	}
    return baseBidOriginal_dict
    
def get_future_base_dic():
    bF_ids = crew_df[(crew_df.Bid_BaseChange == 1) | (crew_df.Bid_BaseChange == 2)]['Crew_ID']
    #print "number of pilots wanting base change is " + str(len(bO_ids))
    baseBid_dict = {
	"Base Bid" : set(bF_ids),
	}
    return baseBid_dict

DEMANDDATA_CSV = 'DemandData.csv'

demand_df = pandas.read_csv(DEMANDDATA_CSV)


def get_demand(base, fleet, rank, week):
	# example: base = "B1", fleet = "A330", rank = "FO", week = 0
	# return the demand at B1, A330, FO of week 0
	demand = demand_df[(demand_df.Weeks == week)][base+'_'+fleet[1:4]+rank]
	return demand

def get_future_position(base, fleet, rank):
	# return non-fix group pilot ids whose future position is input
	return 0

def get_orig_position(base, fleet, rank):
	# return non-fix group pilot ids whose orig position is input
	return 0

def get_position(base, fleet, rank):
	# return fix group pilot ids whose position is input
	return 0
