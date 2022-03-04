import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from Simulator_engine.Farm import Farm
from Simulator_engine.System import System
from Simulator_engine.run_sim import init_run_sim
import time


#========= Farms of interest ===============
farm_names =['Farm_1','Farm_2','Farm_3','Farm_4']
farm_start =[0,100,200,300]
#====== Fá loopið at koyra sum tað skal ====
inital_start = 0
delta_time = 0.5  #
stop_time = 1000
#======= make matrix to days of year =======
Con_matrix = np.zeros((len(farm_names),len(farm_names),366,40))
interanl_con = 0.005 #0.02
external_con = 0.0000
delay = 5
for index,i in enumerate(farm_names):
    Con_matrix = Con_matrix+external_con
    Con_matrix[index,index,:,delay] = interanl_con

#for i in range(0,365):
#    Run_matrix=np.dstack((Run_matrix, c_matrix))
#    Run_matrix_age = np.dstack((Run_matrix_age, c_matrix_age))

#=========== define treatments ==============
treat_date = [[0,200],[0],[0],[0]]
num_treatment = [len(x) for x in treat_date]
Treatment_array = [np.zeros((6, x)) for x in num_treatment]
Treatment_array = [x+0.1 for x in Treatment_array]
Treatment_array[3] = Treatment_array[2]*0+0.95
treatment_type =[['any','Slice'],['any'],['any'],['any']]#[['any'],['any'],['FoodTreatment:'],['any']]

Treatment_array[0][0:2,0] = Treatment_array[0][0:2,0]*0+0.11 # chalimus
Treatment_array[0][2:4,0] = Treatment_array[0][2:4,0]*0+0.33 # Pre-adult
Treatment_array[0][4:6,0] = Treatment_array[0][4:6,0]*0+0.55 # Adult female
#====================== manually determine effect of food treatments =====================
idx_slice = []
for x in ['Slice', 'Emamectin', 'FoodTreatment:']:
    for index,treat_farm in enumerate(treatment_type):
        idx_treat = np.where(np.isin(treat_farm,x))[0]
        if len(idx_treat)>0:
            idx_slice.append([index,idx_treat])

for slices in idx_slice:
    Treatment_array[slices[0]][0][slices[1]] = 0.00  # Chalimus 1
    Treatment_array[slices[0]][1][slices[1]] = 0.00  # Chalimus 2
    Treatment_array[slices[0]][2][slices[1]] = 0  # Pre-adult 1
    Treatment_array[slices[0]][3][slices[1]] = 0  # Pre-adult 2
    Treatment_array[slices[0]][4][slices[1]] = 0  # adult
    Treatment_array[slices[0]][5][slices[1]] = 0  # adult gravid

#======= initiate farm class =============
wrasse_data =[[0, 0],[0, 0]],\
             [[0, 0],[0, 0]],\
             [[0, 0],[0, 0]],\
             [[0, 0],[0, 0]] # [[0],[0]],[[0],[0]],[[0],[0]],[[100,200 ],[50_000,50_000]]

#============== temperature ==================
date = np.arange(0, stop_time)
temp = np.sin(date / (365 / (np.pi * 2))+450) * 2 + 8.5

#temp = temp*0+10
plt.plot(date,temp)
#plt.show()
temperature = [date, temp]

farms = [
Farm(
     delta_time,
     1_000_000,
     L_0=300,                                                             # number of larvea per day
     name=farm_names[index],
     farm_start=farm_start[index]-inital_start,
     prod_len = [400,400],                                                      # production lenght
     fallow = [60,60],
     treatments = np.array(treat_date[index]),                              # Inputs the dates treatments are preformed
     treatment_type = treatment_type[index],                                              # Inputs type of treatments
     treat_eff=np.array(Treatment_array[index]),
     fish_count_history = [np.arange(0,1000), np.arange(0,1000)*0+500_000],   # date, number of fish here set to 500_000 fish
     temperature = temperature,
     CF_data = np.array(wrasse_data[index]),
     biomass_data = [[0,stop_time],np.array([1,1])],
     initial_start=inital_start,
     cleanEff =0.3,
     lice_mortality=[0.01,0.01,0.02,0.02,0.02,0.02],
     surface_ratio_switch=0, # 0 = not used, 1 = constant receiving, 2 = functional response II, 3 = dependant on bio mass
     surface_ratio_k=0.15,
     treatment_period = 10
    )
for index,farm_id in enumerate(farm_names)
]

#c_matrix =  Run_matrix
#Run_matrix_age[pd.isna(Run_matrix_age)]=0
#c_matrix_age = Run_matrix_age

system = System(farms=farms, c_matrix=Con_matrix)

start = time.time()
print("engine start")

koyringar =[]

koyringar.append((delta_time,inital_start, stop_time, system))

# ========== main loop======================
Sim_out = init_run_sim(koyringar[0])

end = time.time()
print(end - start)
#================= plot stuff ==============
time = Sim_out.Farm_0.Date



def plot_farm_all_stages(farm_name):

    Farm_data = Sim_out.__dict__[farm_name]
    sessile_lice = Farm_data.CH_1_f + Farm_data.CH_2_f + Farm_data.CH_1_m + Farm_data.CH_2_m
    PAAM_lice = Farm_data.PA_1_f + Farm_data.PA_2_f + Farm_data.PA_1_m + Farm_data.PA_2_m + Farm_data.AM
    AF_lice = Farm_data.AF
    AF_gravid_lice = Farm_data.AF_gravid
    fig, ax = plt.subplots()

    ax.plot(time, sessile_lice, 'g-', label='Chalimus', linewidth=3)
    ax.plot(time, PAAM_lice, 'y-', label='PA', linewidth=3)
    ax.plot(time, AF_lice+AF_gravid_lice, 'r-', label='Kynsbúnar kvennlús', linewidth=3)
    #ax.plot(time, AF_gravid_lice, 'k-', label='AF$_{egg}$', linewidth=3)
    ax.set_ylabel('lice/salmon', fontsize=20)
    ax.set_xlabel('time (days since stocking)', fontsize=20)

    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1000)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.legend(loc="upper left", fontsize=15)

    return


def plot_all_farms():

    idx_farms = np.flatnonzero(np.core.defchararray.find(Sim_out.__dir__(), "Farm") != -1)

    fig, ax = plt.subplots()

    for idx in idx_farms:
        farm_names = Sim_out.__dir__()[idx]
        Farm_data = Sim_out.__dict__[farm_names]
        sessile_lice = Farm_data.CH_1_f + Farm_data.CH_2_f + Farm_data.CH_1_m + Farm_data.CH_2_m
        PAAM_lice = Farm_data.PA_1_f + Farm_data.PA_2_f + Farm_data.PA_1_m + Farm_data.PA_2_m + Farm_data.AM
        AF_lice = Farm_data.AF
        AF_gravid_lice = Farm_data.AF_gravid




        #ax.plot(time, sessile_lice, 'g-', label='Chalimus', linewidth=3)

        #ax.plot(time, PAAM_lice, 'y-', label='PA', linewidth=3)
        ax.plot(time, AF_lice+AF_gravid_lice, '-', label=farm_names, linewidth=3)

        ax.set_ylabel('lice/salmon', fontsize=20)
        ax.set_xlabel('time (days since stocking)', fontsize=20)

        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1000)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)

        ax.legend(loc="upper left", fontsize=15)
    plt.show()
    return


plot_farm_all_stages("Farm_0")
plot_all_farms()

manager = plt.get_current_fig_manager()
#manager.window.showMaximized()
#plt.tight_layout()

#fig.savefig('Figure_treatments_temp_10_4.pdf')

fig, ax = plt.subplots()

ax.plot(x,y_AF_farm[0],'r-',linewidth = 3)
ax.plot(x,y_AF_farm[1],'r-',linewidth = 3)
ax.plot(x,y_AF_farm[2],'r-',linewidth = 3)
ax.plot(x,y_AF_farm[3],'r-',linewidth = 3)
#ax[0,1].plot(x,y_AG_farm[1],'k-',linewidth = 3)
ax.set_ylim(0,10)
ax.set_xlim(0,1000)

fig3, ax3 = plt.subplots()

#ax3.plot(x,y_Ch_farm[0],'r-',linewidth = 3)
#ax3.plot(x,y_Ch_farm[1],'r-',linewidth = 3)
ax3.plot(x,y_Ch_farm[2],'r-',linewidth = 3)
#ax3.plot(x,y_Ch_farm[3],'r-',linewidth = 3)
#ax[0,1].plot(x,y_AG_farm[1],'k-',linewidth = 3)

ax3.set_xlim(0,1000)


plt.show()


