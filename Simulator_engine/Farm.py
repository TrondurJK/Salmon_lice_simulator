from .Lice_agent_female import Lice_agent_f
from .Lice_agent_male import Lice_agent_m
import numpy as np
from scipy.interpolate import interp1d
from .Planktonic_agent import Planktonic_agent
import pandas as pd
from matplotlib import dates

class Farm:
    '''
    Ein klassi til at fylgja við støðuni á teimum einstøku farminum
    '''
    count = 1  #  ein countari til at hjálpa við at geva farminum navn
    treat_id = 1
    def __init__(self, time,delta_time, fish_count = 1_000_000, L_0=0.2, name=None, farm_start=0,
                 prod_len=420_000, fallow=10000, prod_cyc = 0, treatments=None,
                 treatment_type = None, NumTreat=0, treat_eff=np.array([[]]),
                 weight = 0.2, lusateljingar=[], fish_count_history = None,temperature=None,
                 temperature_Average=None,wrasse_data =None,biomass_data =None,initial_start=0,
                 cleanEff =None,lice_mortality=None,surface_ratio_switch=None):
        '''
        :params:
            time            Tíðin tá ið farmin verður gjørd
            fish_count      Hvussu nógvur fiskur er í farmini tá ið hon startar (count)
            L_0             Extern smitta, (Lús per dag per fisk)
            name            Hvat er navni á farmini um hettar ikki er sett verður tað sett fyri ein
            farm_start      Nær startar productiónin (dagar frá t=0)
            prod_len        Hvussu langur er hvør syklus (dagar)
            fallow          Brakkleggjing (Hvussu leingi millum hvørja útsetan) (dagar)
            prod_cyc        Hvat fyri cycul byrja vit vit við
            treatments      Ein listi av datoion av treatments
            treatment_type  Hvat fyri treatment verður brúkt
            NumTreat        Hvat fyri treatment starta vit við
            treat_eff       Ein matrix av treatment effiicies
            lusateljingar   Ein listi sum sigur nar vit hava lúsateljingar
        '''
        self.time = time


        self.delta_time = delta_time
        #self.temperature = temperature
        self.fish_count = fish_count
        self.fish_count_history = fish_count_history
        self.lice_f = []
        self.lice_m = []
        #self.plankton = np.array([0, 0])
        #  TODO hettar skal verða eitt anna slag av lús
        self.lice_mortality = lice_mortality
        self.adultlice_f = Lice_agent_f(time, 0, 1000,self.lice_mortality) # Ja eg skilji ikki orduliga hi tú setur hettar til 1000
        self.adultlice_m = Lice_agent_m(time, 0, 1000,self.lice_mortality)

        self.L_0 = L_0
        # Hvissi einki navn er sett so gerða vit eitt
        if name == None:
            self.name = 'farm_%s' % self.__class__.count
            self.__class__.count += 1
        else:
            self.name = name
        self.prod_len = prod_len
        self.fallow = fallow
        self.prod_time = -farm_start
        self.__fordeiling__ = [0, 0, 0, 0, 0,0,0,0,0,0]
        self.prod_cyc = prod_cyc
        self.treatment = treatments
        self.SliceON =-1 # -1 merkir at Slice ella Foodtreatment aðrar treatments ikk eru ON. >= 0 eru ON
        self.NumTreat = NumTreat
        self.treatment_type = treatment_type
        self.treat_eff = treat_eff
        self.num_treat_tjek = np.alen(treatments)-1
        dofy = np.arange(0, 366)
        diff_treat = np.append(dofy[0:int(len(dofy)/2)]*0+20,dofy[int(len(dofy)/2):]*0+80)
        self.diff_treatment = [dofy,diff_treat]
        self.num_of_treatments = 0

        self.prod_len_tjek = len(self.prod_len)
        self.weight = weight
        self.W0 = 7 # maximum kg av laksinum
        self.K = 0.008  # growth rate í procent pr dag
        self.old_fish_count = 0
        self.fish_count_update = interp1d(
            x = fish_count_history[0], # remember which is which this should be date of fish
            y = fish_count_history[1], # remember which is which this should be number of fish
            bounds_error = False,
            fill_value = 0
        )
        self.Temp_update = interp1d(
            x = temperature[0], # remember which is which this should be date of fish
            y = temperature[1], # remember which is which this should be number of fish
            bounds_error = False,
            fill_value = 0
        )
#        self.Temp_update_average = interp1d(
#            x = temperature_Average.day_of_year.values, # remember which is which this should be date of fish
#            y = temperature_Average.Temp.values, # remember which is which this should be number of fish
#            bounds_error = False,
#            fill_value = 0
#        )
        self.wrasse_data = wrasse_data
        #self.cleaner_count_update = interp1d(
        #    x = wrasse_data[0], # remember which is which this should be date of fish
        #    y = wrasse_data[1], # remember which is which this should be number of fish
        #    bounds_error = False,
        #    fill_value = 0
        #)
        self.cleaner_fish = 0
        self.cleanEff = cleanEff
        self.biomass = biomass_data
        self.biomass_update = interp1d(
            x = biomass_data[0], # remember which is which this should be date of fish
            y = biomass_data[1]*0.001, # remember which is which this should be number of fish
            bounds_error = False,
            fill_value = 0
        )
        self.plankton = Planktonic_agent(0, self.delta_time)
        self.initial_start = initial_start
        self.surface_ratio_switch = surface_ratio_switch

    def update(self, attached=0):
        '''
        Hvat skal henda tá ið man uppdaterar Farmina
        :params:
        delta_time      Hvussu nógv skal tíðin flyta seg
        attached        Hvussu nógv lús kemur frá ørðum farms
        '''

        self.time += self.delta_time


        #print(self.temp)

        #  TODO hettar við start tíð skal sikkurt formilerðast við at gerða
        #  TODO  self.prod_time negativft í __init__
        #   if self.fish_count_gen:
        #   self.fish_count = self.fish_count_genirator.__next__()
        self.old_fish_count = self.fish_count
        self.fish_count = self.fish_count_update(self.time)
        # attchedment_ratio = TODO ger ein attachement ratio  fiskar í byrjandi skulu ikki hava líka nógva lús

        self.temp = self.Temp_update(self.time)
        #print(self.temp)
        #print(self.temp)
        #if self.temp==0:
        time_new = pd.to_datetime(dates.num2date(self.time+self.initial_start))
        dayofyear = time_new.dayofyear
        #  self.temp=self.Temp_update_average(dayofyear)
        #print(self.fish_count)


        #  hvussu nógvur fiskur er deyður relatift
        if self.old_fish_count != 0:
            deth_ratio = min(1, self.fish_count / self.old_fish_count)
        else:
            deth_ratio = 1

        #  ratio between surface area between  nógvur fiskur er deyður relatift
        if self.surface_ratio_switch == 1:
            Volumen_farm = (self.biomass_update(self.time)*self.fish_count)/20 #schooling density
            Volumen_grid = 160*3*160*3*15
            A_farm  =np.sqrt(Volumen_farm / 15)*np.sqrt(Volumen_farm / 15)
            A_grid = 160*3*160*3
            surface_grid = np.sqrt(Volumen_grid / 15)#4 * 15 * np.sqrt(Volumen_grid / 15) #+ 160*3*2
            surface_farm = np.sqrt(Volumen_farm / 15)#4 * 15 * np.sqrt(Volumen_farm / 15) #+ (Volumen_farm / 10)
            #print(self.biomass_update(self.time),self.fish_count)
            self.surface_ratio = surface_farm/surface_grid
            #self.surface_ratio = A_farm / A_grid
            #self.surface_ratio = Volumen_farm / Volumen_grid
        else:
            self.surface_ratio = 1
        #print(surface_ratio)
        #print(self.fish_count)

        if self.prod_cyc < self.prod_len_tjek:
            #self.plankton = Planktonic_agent(0, 0)

            if self.prod_time <= self.prod_len[self.prod_cyc] and self.prod_time >= 0:


                #============ clenar fish effect========
                #self.cleaner_fish = self.cleaner_count_update(self.time)
                self.cleaner_fish += self.cleaner_F_update(self.time)
                #print(self.cleaner_fish)
                #print(self.name,self.time)
                self.cleaner_fish += -self.cleaner_fish*self.delta_time*0.005

                self.cleaner_death = self.cleaner_fish * self.cleanEff *self.delta_time  # how many lice do cleaner fish eat in one delta time step (0.05 per day)
                #print(self.cleaner_fish,self.cleaner_death,self.time,self.cleanEff)
                if self.cleaner_death <=0 or  np.sum(self.get_fordeiling())==0:
                    self.cleaner_death_ratio = 1
                else:
                    self.cleaner_death_ratio = max([0,(1-self.cleaner_death /
                                                       np.sum([np.sum(self.get_fordeiling()[2:6]), np.sum(self.get_fordeiling()[8:])])   )])


                #=====================
                self.prod_time += self.delta_time
                while self.lice_f:
                    if self.lice_f[0].get_stage() == 'Adult_gravid':
                        self.adultlice_f.count += self.lice_f.pop(0).count
                    else:
                        break  # Hví break her??
                self.adultlice_f.update(self.temp, self.delta_time, deth_ratio, self.cleaner_death_ratio)
                while self.lice_m:
                    if self.lice_m[0].get_stage() == 'Adult':
                        self.adultlice_m.count += self.lice_m.pop(0).count
                    else:
                        break  # Hví break her??
                self.adultlice_m.update(self.temp, self.delta_time, deth_ratio,self.cleaner_death_ratio)

                for mylice_f in self.lice_f:
                    mylice_f.update(self.temp, self.delta_time, deth_ratio,self.cleaner_death_ratio)
                for mylice_m in self.lice_m:
                    mylice_m.update(self.temp, self.delta_time, deth_ratio,self.cleaner_death_ratio)
                #  Create a new Lice_agent
                temp_lice_female = Lice_agent_f(self.time, 0, 0,self.lice_mortality)
                temp_lice_male = Lice_agent_m(self.time, 0, 0, self.lice_mortality)
                mu_f = temp_lice_female.get_mu()
                mu_m = temp_lice_male.get_mu()
                self.update_weigh(self.prod_time)
                #print(attached)
                #print(self.time,self.name)
                smitta = (self.L_0 + attached)*self.surface_ratio
                #print(self.surface_ratio)
                #if self.name == 'Lopra':
                #    print(smitta,self.name)

                temp_lice_female.count = smitta/mu_f * (1 - np.exp(-mu_f * self.delta_time))*0.5
                temp_lice_male.count = smitta / mu_m * (1 - np.exp(-mu_m * self.delta_time))*0.5
                self.lice_f.append(temp_lice_female)
                self.lice_m.append(temp_lice_male)
                #print([np.sum(self.get_fordeiling()[0:6]), np.sum(self.get_fordeiling()[6:])])
                #print(self.name,self.time)

            elif self.prod_time >= self.prod_len[self.prod_cyc]:
                #self.get_fordeiling = self.get_fordeiling()*0
                self.fish_count =0
                self.lice_f = []
                self.lice_m = []
                self.plankton = Planktonic_agent(0, self.delta_time)
                self.adultlice_f = Lice_agent_f(self.time, 0, 1000,self.lice_mortality)
                self.adultlice_m = Lice_agent_m(self.time, 0, 1000, self.lice_mortality)
                self.prod_time = -self.fallow[self.prod_cyc]
                self.prod_cyc += 1
                self.cleaner_fish =0


            else:
                self.prod_time += self.delta_time


        #print(self.SliceON)
        if self.NumTreat<=self.num_treat_tjek:
            if self.time > self.treatment[self.NumTreat]: # and self.treatment[self.NumTreat]>0:
                #print(self.treatment_type[self.NumTreat])
                if self.treatment_type[self.NumTreat] == 'FoodTreatment:':
                    self.NumTreat_slice = self.NumTreat
                    self.avlusing('Slice',self.NumTreat_slice, 1, self.temp)
                    self.SliceON = 40
                else:
                    self.avlusing('TreatmentY', self.NumTreat, 1, self.temp)

                self.NumTreat += 1

            elif np.isnan(self.treatment[self.NumTreat]):
                self.NumTreat += 1
        if self.SliceON >= 0:
            self.avlusing('Slice', self.NumTreat_slice, 1, self.temp)

            #print(self.SliceON,self.NumTreat)
            self.SliceON += -self.delta_time

        if self.fish_count!=0:
            #print(self.name,self.get_fordeiling(calculate=True)[4],self.time)
            #print(np.sum([self.get_fordeiling(calculate=True)[4], self.get_fordeiling(calculate=True)[5]]))
            # dayofyear
            relevant_treatment = self.diff_treatment[1][self.diff_treatment[0]==dayofyear]
            if np.sum([self.get_fordeiling(calculate=True)[4:6]]) / self.fish_count > relevant_treatment: # OOurt ther sum sigur nær tað er hvat í løbi av árinum
                self.avlusing('TreatmentX', 1, 1, self.temp)
                self.num_of_treatments += 1

        if self.prod_time<0:
            #print(self.time, 'Fasle',self.get_fordeiling())
            self.get_fordeiling(fallow=1)
        else:
            #print(self.time,'True',self.get_fordeiling())
            self.get_fordeiling(calculate=True)

    def avlusing(self, slag, NumTreat, consentration, temp):
        '''
        Hvat sker tá ið tað verður avlúsa
        :params:
        slag            Hvat fyri slag av avlúsing er talan um
        consentration   Hvussu ógvuslig er avlúsingin
        temp            Hvat er havtempraturin
        -------------------
        slag = 'X' er ein viðgerð sum drepur 95% av øllum føstum lúsum
        '''

        if slag == 'TreatmentY':
            for mylice_f in self.lice_f:
                mylice_f.TreatmentY(self.treat_eff[:, NumTreat])
            for mylice_m in self.lice_m:
                mylice_m.TreatmentY(self.treat_eff[:, NumTreat])
            self.adultlice_f.TreatmentY(self.treat_eff[:, NumTreat])
            self.adultlice_m.TreatmentY(self.treat_eff[:, NumTreat])
        # ==========================================
        elif slag == "TreatmentX":
            for mylice_f in self.lice_f:
                mylice_f.TreatmentX()
            for mylice_m in self.lice_m:
                mylice_m.TreatmentX()
            self.adultlice_f.TreatmentX()
            self.adultlice_m.TreatmentX()

        elif slag == "Slice":
            treat_eff = 1-(1-self.treat_eff[:, NumTreat])*self.delta_time #1-(1-0.95)*1
            for mylice_f in self.lice_f:
                mylice_f.Slice(treat_eff)
            for mylice_m in self.lice_m:
                mylice_m.Slice(treat_eff)
            self.adultlice_f.Slice(treat_eff)
            self.adultlice_m.Slice(treat_eff)
        else:
            raise NotImplementedError

    def get_fordeiling(self, calculate=False,fallow = 0):
        '''
        Gevur ein lista av teimum forskelligu stages
        :params:
        calculate       skal man brúka tíð sístu frodeilingina ella skal hettar roknast
        '''
        #print(calculate,'calcutale',self.time)
        if calculate:
            Ch1_f, Ch2_f, Pa1_f, Pa2_f, Adult_f,Adult_gravid_f, Ch1_m, Ch2_m, Pa1_m, Pa2_m, Adult_m = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            # Chalimus 1, Chalimus 2, Pre-Adult 1, Pre-Adult 2, Adult
            for lus in self.lice_f:
                my_nr = lus.get_stage()
                if my_nr == 'Ch1':
                    Ch1_f += lus.count
                if my_nr == 'Ch2':
                    Ch2_f += lus.count
                if my_nr == 'Pa1':
                    Pa1_f += lus.count
                if my_nr == 'Pa2':
                    Pa2_f += lus.count
                if my_nr == 'Adult':
                    Adult_f += lus.count
                if my_nr == 'Adult_gravid':
                    Adult_gravid_f += lus.count
            Adult_gravid_f += self.adultlice_f.count

            for lus in self.lice_m:
                my_nr = lus.get_stage()
                if my_nr == 'Ch1':
                    Ch1_m += lus.count
                if my_nr == 'Ch2':
                    Ch2_m += lus.count
                if my_nr == 'Pa1':
                    Pa1_m += lus.count
                if my_nr == 'Pa2':
                    Pa2_m += lus.count
                if my_nr == 'Adult':
                    Adult_m += lus.count

            Adult_m += self.adultlice_m.count

            self.__fordeiling__ = [Ch1_f, Ch2_f, Pa1_f, Pa2_f, Adult_f,Adult_gravid_f,Ch1_m, Ch2_m, Pa1_m, Pa2_m, Adult_m]
        elif fallow == 1:
            Ch1_f, Ch2_f, Pa1_f, Pa2_f, Adult_f,Adult_gravid_f, Ch1_m, Ch2_m, Pa1_m, Pa2_m, Adult_m = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            self.__fordeiling__ = [Ch1_f, Ch2_f, Pa1_f, Pa2_f, Adult_f,Adult_gravid_f,Ch1_m, Ch2_m, Pa1_m, Pa2_m, Adult_m]
        return self.__fordeiling__

    #def __repr__(self):
    #    return self.smitta # 'farmin %s sum hevur %s fiskar' % (self.name, self.fish_count)

    def update_weigh(self,prod_time):
        self.W = self.W0*(1-np.exp(-self.K*prod_time))**3  #(1-np.exp(-self.K*(t)))**3 #0.0028 * self.W ** (2 / 3) * (1 - (self.W / 6) ** (1 / 3))
        return self.W


    def cleaner_F_update(self,time):
        idx = np.where(np.logical_and(self.wrasse_data[0] > time - 0.001, self.wrasse_data[0] < time + 0.001))
        if len(idx[0])>0:
            return np.sum(self.wrasse_data[1][idx[0][:]])
        else:
            return 0