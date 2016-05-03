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


##########  Draw shortage and surplus ##############

shortage_df = pd. read_csv('shortage_28.csv', index_col= 0)
shortage_df.columns=['base','position','time','value']

#surplus_df = pd. read_csv('surplus_n.csv', index_col = 0)
#surplus_df.columns=['base','position','time','value']


###### base 1 #######
CPT_A320 = list(shortage_df[(shortage_df.position =='CPTA320')&(shortage_df.base == 1)]['value'])
CPT_A330 = list(shortage_df[(shortage_df.position =='CPTA330')&(shortage_df.base == 1)]['value'])
FO_A320 = list(shortage_df[(shortage_df.position =='FOA320')&(shortage_df.base == 1)]['value'])
FO_A330 = list(shortage_df[(shortage_df.position =='FOA330')&(shortage_df.base == 1)]['value'])



# plot shortage

x = list(range(1,27))

fig, ax = plt.subplots(2, sharex=True, figsize=(90,20))
N = 26
ind = np.arange(N)

width=0.3
#fig = plt.figure()
#ax = fig.add_subplot(111)
rec1 = ax[0].bar(ind-width, CPT_A320,width,color='b',align='center')
rec2 = ax[0].bar(ind, CPT_A330,width,color='g',align='center')
rec3 = ax[0].bar(ind+width, FO_A320,width,color='r',align='center')
rec4 = ax[0].bar(ind+width*2, FO_A330,width,color='pink',align='center')

ax[0].set_ylabel('shortage')
ax[0].set_xticks(ind+width/2)
ax[0].set_xticklabels(x)
ax[0].legend((rec1[0],rec2[1],rec3[2],rec4[3]),('CPT_A320','CPT_A330','FO_A320','FO_A330'),loc='best')


ax[0].autoscale(tight=True)
ax[0].set_title('Base 1 ', fontsize=14)



###### base2 ##########

CPT_A320 = list(shortage_df[(shortage_df.position =='CPTA320')&(shortage_df.base == 2)]['value'])
CPT_A330 = list(shortage_df[(shortage_df.position =='CPTA330')&(shortage_df.base == 2)]['value'])
FO_A320 = list(shortage_df[(shortage_df.position =='FOA320')&(shortage_df.base == 2)]['value'])
FO_A330 = list(shortage_df[(shortage_df.position =='FOA330')&(shortage_df.base == 2)]['value'])




rec1 = ax[1].bar(ind-width, CPT_A320,width,color='b',align='center')
rec2 = ax[1].bar(ind, CPT_A330,width,color='g',align='center')
rec3 = ax[1].bar(ind+width, FO_A320,width,color='r',align='center')
rec4 = ax[1].bar(ind+width*2, FO_A330,width,color='pink',align='center')

ax[1].set_ylabel('shortage & surplus')
ax[1].set_xticks(ind+width/2)
ax[1].set_xticklabels(x)
ax[1].legend((rec1[0],rec2[1],rec3[2],rec4[3]),('CPT_A320','CPT_A330','FO_A320','FO_A330'),loc='best')


ax[1].autoscale(tight=True)
ax[1].set_title('Base 2', fontsize=14)



plt.show()





plt.savefig('shortage.png')

##########  Draw pilot status heatmap ##############

status_df = pd.read_csv('status_28.csv')



#row_labels=list(status_df.columns)
#column_labels = list(status_df.index)


data =  pd.read_csv('status_28.csv',index_col=0)
width = len(data.columns)/7*10
height = len(data.index)/7*10
fig, ax = plt.subplots(figsize=(width,height))

cMap = ListedColormap(['mintcream', 'lime', 'darkgreen','tomato','darkmagenta','plum','b','cornflowerblue'])
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


#ax.grid(True)



for t in ax.xaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False
for t in ax.yaxis.get_major_ticks():
    t.tick1On = False
    t.tick2On = False


plt.subplots_adjust(top=0.9)
fig.suptitle('Pilots Status in 26 Weeks')
plt.show()





plt.savefig('heatmap_status.png')









