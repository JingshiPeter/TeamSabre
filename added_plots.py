
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
transition_list = []

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
# b1 = 4 b2 = 5 'CPT'=6 'FO'=7
for (p, r, f, b) in model.nonfix_var_set:
	for t in model.time:
		if((p in model.base_pilots) & (t in model.timestart) & ((p,r,f,b) in model.from_pos)):
			if((model.Y[p, r, f, b, t].value == 1) & (model.Y[p, r, f, b, t+1].value == 0)):
				#transition_list.append((p, t, 'b'+str(b)))
				if b == 1:
					status_df.ix[p, t] = 4
				if b == 2:
					status_df.ix[p, t] = 5

for (p, r, f, b) in model.nonfix_var_set:
	for t in model.time:
		if((p in model.rank_pilots) & (t in model.timestart) & ((p,r,f,b) in model.from_pos)):
			if((model.Y[p, r, f, b, t].value == 1) & (model.Y[p, r, f, b, t+1].value == 0)):
				#transition_list.append((p, t, str(r)))
				if r == 'CPT':
					status_df.ix[p, t] = 6
				if r == 'FO':
					status_df.ix[p, t] = 7

# csv
# status_df.to_csv("status.csv",index = True)




# Shortage

shortage_list =[]
surplus_list =[]

for b in model.base:
	for r in model.rank:
		for f in model.fleet:
			for t in model.time:
				if model.shortage[r,f,b,t].value > 0:
					shortage_list.append((b,str(r)+str(f),t,-model.shortage[r,f,b,t].value))
				if model.shortage[r,f,b,t].value == 0:
					shortage_list.append((b,str(r)+str(f),t,0))

				
for b in model.base:
	for r in model.rank:
		for f in model.fleet:
			for t in model.time:
					surplus_list.append((b,str(r)+str(f),t,model.surplus[r,f,b,t].value))
				

shortage = pd.DataFrame(shortage_list)
surplus = pd.DataFrame(surplus_list)

# csv
# shortage.to_csv("shortage.csv",index = True)
# surplus.to_csv("surplus.csv",index = True)

##########  Draw shortage and surplus ##############

shortage_df = pd. read_csv('shortage.csv', index_col= 0)
surplus_df = pd. read_csv('surplus.csv', index_col = 0)

shortage_df['position'] = shortage_df[['rank','fleet']].apply(lambda x: '_'.join(x), axis=1)
shortage_df['position'] = shortage_df['base'].map(str)+shortage_df['position']

b1CPTA330 = shortage_df[(shortage_df.position =='1CPT_A330')][['value','time']]
b1FOA320 = shortage_df[(shortage_df.position =='1FO_A320')][['value','time']]
b1FOA330 = shortage_df[(shortage_df.position =='1FO_A330')][['value','time']]
b2FOA330 = shortage_df[(shortage_df.position =='2FO_A330')][['value','time']]



# plot shortage

x = list(range(1,27))

b1_CPT_A330 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1]
b1_FO_A320  = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,2,2,2,2,2,2,2,2]
b1_FO_A330  = list(b1FOA330['value'])
b2_FO_A330  = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1]
fig, ax = plt.subplots(figsize=(90,20))
N = 26
ind = np.arange(N)

width=0.3
#fig = plt.figure()
#ax = fig.add_subplot(111)
rec1 = ax.bar(ind-width, b1_CPT_A330,width,color='b',align='center')
rec2 = ax.bar(ind, b1_FO_A320,width,color='g',align='center')
rec3 = ax.bar(ind+width, b1_FO_A330,width,color='r',align='center')
rec4 = ax.bar(ind+width*2, b1_FO_A330,width,color='pink',align='center')

ax.set_ylabel('shortage')
ax.set_xticks(ind+width/2)
ax.set_xticklabels(x)
ax.legend((rec1[0],rec2[1],rec3[2],rec4[3]),('b1_CPT_A330','b1_FO_A320','b1_FO_A330','b2_FO_A330'),loc='best')
#def autolabel(rec):
  #  for rect in rec:
    #    h = rect.get_height()
     #  ax.text(rect.get_x()+rect.get_width()/2., 0.08*h, '%d'%int(h),
          #      ha='center', va='bottom')

#autolabel(rec1)
#autolabel(rec2)
#autolabel(rec3)
ax.autoscale(tight=True)

plt.show()

    ####def equalizeLists(*lists):
    #maxLen = max([len(list) for list in lists])
    #for list in lists:
      # list = list.extend([0]*(maxLen - len(list)))
    #return maxLen
#This will equalize the lengths of two or more lists automatically by adding zeros to the ends of the shorter ones. You could insert it into your code like so:

         



plt.savefig('shortage.png')

##########  Draw pilot status heatmap ##############

status_df = pd.read_csv('status.csv')



#row_labels=list(status_df.columns)
#column_labels = list(status_df.index)


data =  pd.read_csv('status.csv',index_col=0)
width = len(data.columns)/7*10
height = len(data.index)/7*10
fig, ax = plt.subplots(figsize=(width,height))

cMap = ListedColormap(['mintcream', 'lime', 'darkgreen','tomato','darkmagenta','plum','b','royalblue'])
heatmap = ax.pcolor(data, cmap =cMap)

#heatmap = ax.pcolor(data, cmap ="Pastel2")
#heatmap = ax.pcolor(data, cmap=plt.cm.Blues, alpha=0.8) 

#legend
# Turn off all the ticks
ax = plt.gca()
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad=0.005)

cbar = plt.colorbar(heatmap, cax = cax)

cbar.ax.get_yaxis().set_ticks([])
for j, lab in enumerate(['Work','Trainer','Trainee','Vacation','b1','b2','CPT','FO']):
    cbar.ax.text(.5, (1.45 * j + 1) / 12.0, lab, ha='center', va='center', rotation=270)
cbar.ax.get_yaxis().labelpad =15
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


ax.grid(False)



for t in ax.xaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False
for t in ax.yaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False

plt.show()





#plt.savefig('test.png')








