

# Extract Data

# pilots status

## variables

# model.Yall
# model.T(p, b, t) ==1
# model.Trainee(p, b, t) == 1
# model.V(p, t) == 1

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as pltimport 
import pandas as pd
import csv
import numpy as np


# create new dataframe

# columns =['pilots_id', 'status', 'trasition', 'time', 'time_vacation']
columns =[ '1', '2', '3', '4','5','6','7','8','9','10','11','12','13','14','15','16',
'7','18','19','20','21','22','23','24','25','26']
status_df = pd.DataFrame(index = crew_df['Crew_ID'], columns = columns)

status_df = status_df.fillna(0)

#status_df['pilots_id'] = crew_df['Crew_ID']


trainer_list = []
trainee_list = []
vacation_list = []

# Trainer = 1

for p in model.trainer_pilots:
	for t in model.timestart:
		for b in model.base:
			if model.T[p, b, t].value == 1 :
				#trainer_list.append((p, t, b))
				#status_df.set_value(status_df.pilots_id == p,'status',1)
				#status_df.set_value(status_df.pilots_id == p,'time',t)
				#status_df.set_value(status_df.pilots_id == p, status_df.columns.values[t] == str(t) ,1)
				status_df.ix[p, t]= 1




#trainer = pd.DataFrame(trainer_list)
#trainer.columns = ['ID','Time','Base']

# Trainee = 2
for p in model.fleet_pilots:
	for t in model.timestart:
 		for b in model.base:
 			if model.Trainee[p, b, t].value == 1 :
 				#trainee_list.append((p, t, b))
 				#status_df.set_value(status_df.pilots_id == p,'status',2)
				#status_df.set_value(status_df.pilots_id == p,'time',t)
				#status_df.set_value(status_df.pilots_id == p, status_df.columns.values ==t ,2)
				status_df.ix[p, t]= 2

#trainee = pd.DataFrame(trainee_list)
#trainee.columns = ['ID','Time','Base']

# Vacation = 3

for p in model.pilots:
	for t in model.timestart:
		if model.V[p, t].value == 1:
			#vacation_list.append((p, t))
			#status_df.set_value(status_df.pilots_id == p,'status_vacation',3)
			#status_df.set_value(status_df.pilots_id == p,'time_vacation',t)

			#status_df.set_value(p, t, 3)
			status_df.ix[p, t] = 3


#vacation = pd.DataFrame(vacation_list)
#vacation.columns = ['ID','Time']


# Trasition

transition_list = []
for (p, r, f, b) in model.nonfix_var_set:
	for t in model.time:
		if((p in model.base_pilots) & (t in model.timestart) & ((p,r,f,b) in model.from_pos)):
			if((model.Y[p, r, f, b, t].value == 1) & (model.Y[p, r, f, b, t+1].value == 0)):
				transition_list.append((p, t, 'b'+str(b)))
		if((p in model.rank_pilots) & (t in model.timestart) & ((p,r,f,b) in model.from_pos)):
			if((model.Y[p, r, f, b, t].value == 1) & (model.Y[p, r, f, b, t+1].value == 0)):
				transition_list.append((p, t, 'r'+str(r)))


# csv
# status_df.to_csv("status.csv",index = True)
# copy file to local
# scp -c blowfish -r js79735@me-dimitrovresearch.engr.utexas.edu:/Teamsabre/status.csv /home/user/Desktop/



# heatmap

plt.pcolor(status_df)
plt.colorbar()
fig = plt.figure().add_subplot(111)
fig.set_xticklabels(status_df.columns)
fig.set_yticklabels(status_df.index)

plt.savefig('test.png')








