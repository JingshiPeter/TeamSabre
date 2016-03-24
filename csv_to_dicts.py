import pandas

#DEFINE GLOBAL NAMES HERE
CREWDATA_CSV = 'CrewData.csv'

crew_df = pandas.read_csv(CREWDATA_CSV)

print crew_df.columns

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
	    "cpt" : set(cpt_ids),
	    "fo" : set(fo_ids),
	    }
	print rank_dic

def get_fleet_dic():
	# return a fleet dictionary
	# example :
	# fleets = {

	# 'a330' : {'900201', '900421'},
	# 'a320' : {'900488', '900424'}

	# }  ctrl + / is convert # into real. 
	A320_ids = crew_df[crew_df.Cur_Fleet == "A320"]['Crew_ID']
	A330_ids = crew_df[crew_df.Cur_Fleet == "A330"]['Crew_ID']
	print "number of pilots for A320 is " + str(len(A320_ids))
	print "number of pilots for A330 is " + str(len(A330_ids))
	fleet_dic = {
	    "A320" : set(A320_ids),
	    "A330" : set(A330_ids),
	    }
	print fleet_dic

#def get_from_fleet_dic():

#def get_to_fleet_dic():

#def get_from_rank_dic():

#def get_to_rank_dic():

#def get_from_base_dic():

<<<<<<< Updated upstream
def get_to_base_dic():
=======
#def get_to_base_dic():
	
>>>>>>> Stashed changes
