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

	# }

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
	    "cpt" : set(cpt_ids),
	    "fo" : set(fo_ids)
	    }
	return rank_dic

def get_fleet_dic():
	# return a fleet dictionary
	# example :
	# fleets = {

	# 'a330' : {'900201', '900421'},
	# 'a320' : {'900488', '900424'}

	# }
def get_from_fleet_dic():

def get_to_fleet_dic():

def get_from_rank_dic():

def get_to_rank_dic():

def get_from_base_dic():

def get_to_base_dic():
