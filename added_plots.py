

# Extract Data

# pilots status

## variables

# model.Yall
# model.T(p, b, t) ==1
# model.Trainee(p, b, t) == 1
# model.V(p, t) == 1

#import matplotlib for research computer
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pylab import *
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



##########  Draw pilot status heatmap ##############

status_df = pd.read_csv('status.csv')



#row_labels=list(status_df.columns)
#column_labels = list(status_df.index)


data =  pd.read_csv('status.csv',index_col=0)
width = len(data.columns)/7*10
height = len(data.index)/7*10
fig, ax = plt.subplots(figsize=(width,height))

#cMap = ListedColormap(['white', 'green', 'red','blue'])

#heatmap = ax.pcolor(data, cmap ="Pastel2")
heatmap = ax.pcolor(data, cmap=plt.cm.Blues, alpha=0.8) 

#legend
# Turn off all the ticks
ax = plt.gca()
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad=0.005)

cbar = plt.colorbar(heatmap, cax = cax)

cbar.ax.get_yaxis().set_ticks([])
for j, lab in enumerate(['Idle','Trainer','Trainee','Vacation']):
    cbar.ax.text(.5, (2 * j + 1) / 8.0, lab, ha='center', va='center', rotation=270)
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('status', rotation=270)


# Format
fig = plt.gcf()


# turn off the frame
ax.set_frame_on(False)
#ax.set_aspect('equal') 
# put the major ticks at the middle of each cell
ax.set_xticks(np.arange(data.shape[1])+0.5, minor=False)
ax.set_yticks(np.arange(data.shape[0])+0.5, minor=False)

# want a more natural, table-like display
ax.invert_yaxis()
ax.xaxis.tick_top()

#ax.set_xticklabels(data.columns, minor=False)
#ax.set_yticklabels(data.index, minor=False)

ax.set_yticks(np.arange(len(data.index)) + 0.5)
ax.set_yticklabels(data.index, size=8)
ax.set_xticks(np.arange(len(data.columns)) + 0.5)
ax.set_xticklabels(data.columns, size= 10)


#ax.grid(False)



for t in ax.xaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False
for t in ax.yaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False

plt.show()





#plt.savefig('test.png')








