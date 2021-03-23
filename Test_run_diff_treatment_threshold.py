import numpy as np
import pandas as pd
import matplotlib
print('hey hey')
import matplotlib.pyplot as plt
#from Lice_similator.kodur.tictoc import tic, toc
from Simulator_engine.Farm import Farm
from Simulator_engine.System_3_FO import System
from Simulator_engine.run_sim_test import init_run_sim
#from Lice_similator.kodur.Licedata_input_norge import Licedata_NO


#========= Farms of interest ===============
farm_names =['farm 1','farm 2','farm 3','farm 4']
farm_start =[0,100,200,300]
#====== Fá loopið at koyra sum tað skal ====
inital_start = 0
delta_time = 0.5  #
stop_time = 1700
#======= make matrix to days of year =======
Con_matrix = np.zeros((len(farm_names),len(farm_names),366,40))
interanl_con = 0.02
external_con = 0.01
delay = 5
Con_matrix[:, :, :, delay] = external_con
for index,i in enumerate(farm_names):
    Con_matrix[index,index,:,delay] = interanl_con

#for i in range(0,365):
#    Run_matrix=np.dstack((Run_matrix, c_matrix))
#    Run_matrix_age = np.dstack((Run_matrix_age, c_matrix_age))

#=========== define treatments ==============
treat_date = [[0],[0],[0],[0]]
num_treatment = [len(x) for x in treat_date]
Treatment_array = [np.zeros((6, x)) for x in num_treatment]
Treatment_array = [x+0.1 for x in Treatment_array]
Treatment_array[2] = Treatment_array[2]*0+0.95
treatment_type =[['any'],['any'],['any'],['any']]#[['any'],['any'],['FoodTreatment:'],['any']]

#======= initiate farm class =============
wrasse_data =[[0],[0]],[[0],[0]],[[0],[0]],[[0],[0]] # [[0],[0]],[[0],[0]],[[0],[0]],[[100,200 ],[50_000,50_000]]

#============== temperature ==================
#Temp_input_average_year = pd.read_csv('./Data/Average_Temp_year.csv')
date = np.arange(0, stop_time)
temp = np.sin(date / (365 / (np.pi * 2))+300) * 2 + 8
#plt.plot(date,temp)
#plt.show()
temperature = [date, temp]

farms = [
Farm(0,delta_time,1_000_000,
     L_0=500,                                                             # number of larvea per day
     name=farm_names[index],
     farm_start=farm_start[index]-inital_start,
     prod_len = [400,400,400],                                                      # production lenght
     fallow = [60,60,60],
     treatments = np.array(treat_date[index]),                              # Inputs the dates treatments are preformed
     treatment_type = treatment_type[index],                                              # Inputs type of treatments
     NumTreat=0,
     treat_eff=np.array(Treatment_array[index]),
     fish_count_history = [np.arange(0,stop_time), np.arange(0,stop_time)*0+500_000],   # date, number of fish here set to 500_000 fish
     temperature = temperature,
     wrasse_data = np.array(wrasse_data[index]),
     biomass_data = [[0,stop_time],np.array([1,1])],
     initial_start=inital_start,
     cleanEff =0.3,
     lice_mortality=[0.01,0.01,0.02,0.02,0.02,0.02],
     surface_ratio_switch=0
    )
for index,farm_id in enumerate(farm_names)
]

#c_matrix =  Run_matrix
#Run_matrix_age[pd.isna(Run_matrix_age)]=0
#c_matrix_age = Run_matrix_age

system = System(farms=farms, c_matrix=Con_matrix,
                delta_time=delta_time,temperature_input = [[0,stop_time],[10,10]],
                inital_start=inital_start)

koyringar =[]

koyringar.append((delta_time, stop_time, system, inital_start))

# ========== main loop======================
out = init_run_sim(koyringar[0])

#================= plot stuff ==============
x = out[0]
y_AG_farm = []
y_AF_farm = []
y_PA_farm = []
y_Ch_farm = []
for index,i in enumerate(farm_names):
    #y_AG_farm.append(np.array(out[1][index][5]))
    y_AF_farm.append(np.array(out[1][index][4]) + np.array(out[1][index][5]))
    y_PA_farm.append(np.array(out[1][index][2]) + np.array(out[1][index][3]))
    y_Ch_farm.append(np.array(out[1][index][0]) + np.array(out[1][index][1]))

Nfish = out[2]
fig, ax = plt.subplots(2,2,gridspec_kw = {'wspace':0.1, 'hspace':0.1})
#plt.xticks(xvalues)

ax[0,0].plot(x,y_Ch_farm[0],'g-',label='Chalimus',linewidth = 3)
ax[0,0].plot(x,y_PA_farm[0],'y-',label='PA',linewidth = 3)
ax[0,0].plot(x,y_AF_farm[0],'r-',label='AF',linewidth = 3)
#ax[0,0].plot(x,y_AG_farm[0],'k-',label='AF$_{egg}$',linewidth = 3)
#ax[0,0].set_xlabel('time (days since stocking)',fontsize=15)
ax[0,0].set_ylabel('lice/salmon',fontsize =20)
ax[0,0].set_ylim(0,2)
ax[0,0].set_xlim(0,1700)
#ax2 = ax[0,0].twinx()
ax[0,0].plot(date,temp/10,'b-',label='Temperature/10')
ax[0,0].tick_params(axis='x', labelsize=15)
ax[0,0].tick_params(axis='y', labelsize=15)
#ax[0,0].xaxis.set_visible(False)
ax[0,0].legend(loc ="upper left",fontsize =15)

ax[0,1].plot(x,y_Ch_farm[1],'g-',linewidth = 3)
ax[0,1].plot(x,y_PA_farm[1],'y-',linewidth = 3)
ax[0,1].plot(x,y_AF_farm[1],'r-',linewidth = 3)
#ax[0,1].plot(x,y_AG_farm[1],'k-',linewidth = 3)
#ax[0,1].plot([200,200],[0,max(y_Ch_farm[1])],'k--',label='Treatment: eff. 90%',linewidth = 3)
ax[0,1].set_ylim(0,2)
ax[0,1].set_xlim(0,1700)
#ax[0,1].yaxis.set_visible(False)
#ax[0,1].xaxis.set_visible(False)
ax[0,1].tick_params(axis='x', labelsize=15)
ax[0,1].tick_params(axis='y', labelsize=15)
ax[0,1].legend(loc ="upper left",fontsize =15)

ax[1,0].plot(x,y_Ch_farm[2],'g-',linewidth = 3)
ax[1,0].plot(x,y_PA_farm[2],'y-',linewidth = 3)
ax[1,0].plot(x,y_AF_farm[2],'r-',linewidth = 3)
#ax[1,0].plot(x,y_AG_farm[2],'k-',linewidth = 3)
#ax[1,0].plot([200,200],[0,max(y_Ch_farm[2])],'k--',label='Oral treatment: eff. 5% d$^{-1}$',linewidth = 3)
ax[1,0].set_xlabel('time (days since stocking)',fontsize =20)
ax[1,0].set_ylabel('lice/salmon',fontsize =20)
ax[1,0].set_xlim(0,1700)
ax[1,0].set_ylim(0,2)
#ax[1,1].yaxis.set_visible(False)
#ax[1,1].xaxis.set_visible(False)
ax[1,0].tick_params(axis='x', labelsize=15)
ax[1,0].tick_params(axis='y', labelsize=15)
ax[1,0].legend(loc ="upper left",fontsize =15)

ax[1,1].plot(x,y_Ch_farm[3],'g-',linewidth = 3)
ax[1,1].plot(x,y_PA_farm[3],'y-',linewidth = 3)
ax[1,1].plot(x,y_AF_farm[3],'r-',linewidth = 3)
#ax[1,1].plot(x,y_AG_farm[3],'k-',linewidth = 3)
#ax[1,1].plot([150,150],[0,10],'k--',label='20% CF blend: 0.3 mobile lice fish$^{-1}$ d$^{-1}$',linewidth = 3)
ax[1,1].set_xlabel('time (days since stocking)',fontsize =20)
#ax[1,1].set_ylabel('lice/salmon',fontsize=15)
ax[1,1].set_ylim(0,2)
ax[1,1].set_xlim(0,1700)
#ax[1,1].yaxis.set_visible(False)
#ax[1,1].xaxis.set_visible(False)
ax[1,1].tick_params(axis='x', labelsize=15)
ax[1,1].tick_params(axis='y', labelsize=15)
ax[1,1].legend(loc ="upper left",fontsize =15)

manager = plt.get_current_fig_manager()
manager.window.showMaximized()
#plt.tight_layout()
total_num_treat = np.sum([out[6][0][-1:],out[6][1][-1:],out[6][2][-1:],out[6][3][-1:]])
print(total_num_treat)


plt.show()
fig.savefig('Figure_diff_treatment_TT_05.pdf')

plt.show()





#plt.plot(x,y_AF_farm_2)
print('kemur tað her til?')

