from .Lice_agent import Lice_agent_f, Lice_agent_m
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
                 temperature_Average=None,CF_data =None,biomass_data =None,initial_start=None,
                 cleanEff =None,lice_mortality=None,surface_ratio_switch=False,
                 use_cleaner_F_update=False, seasonal_treatment_treashold=False,
                 treatment_period = 1):
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
        self.fish_count = fish_count
        self.fish_count_history = fish_count_history
        self.lice_f = {
            'Ch1': [],
            'Ch2': [],
            'Pa1': [],
            'Pa2': [],
            'Adult': [],
        }

        self.lice_m = {
            'Ch1': [],
            'Ch2': [],
            'Pa1': [],
            'Pa2': [],
        }


        #self.plankton = np.array([0, 0])
        #  TODO hettar skal verða eitt anna slag av lús


        self.lice_mortality = lice_mortality
        if len(lice_mortality) >= 6:
            pass
        else:
            raise Exception('lice_mortality skal minst verða 6 langur')
        self.lice_mortality_dict = {slag:mortality 
                                    for slag, mortality in zip(
                                        ['Ch1', 'Ch2', 'Pa1', 'Pa2', 'Adult', 'Adult_gravid'],
                                        lice_mortality
                                    )
                                   }

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
        self.__fordeiling__ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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
        self.treatment_period = treatment_period
        self.prod_len_tjek = len(self.prod_len)
        self.done = False
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

        if temperature_Average is None:
            self.Temp_update = interp1d(
                x = temperature[0], # remember which is which this should be date of fish
                y = temperature[1], # remember which is which this should be number of fish
                bounds_error = False,
                fill_value = 0
            )
            self.Temp_update_average = None
        else:
            self.Temp_update_average = interp1d(
                x = temperature_Average.day_of_year.values, # remember which is which this should be date of fish
                y = temperature_Average.Temp.values, # remember which is which this should be number of fish
                bounds_error = False,
                fill_value = 0
            )
        self.CF_data = CF_data
        self.use_cleaner_F_update = use_cleaner_F_update
        if self.CF_data is not None:
            self.cleaner_count_update = interp1d(
                x = CF_data[0], # remember which is which this should be date of fish
                y = CF_data[1], # remember which is which this should be number of fish
                bounds_error = False,
                fill_value = 0
            )
        self.cleaner_fish = 0
        #  TODO what to do with cleanEff if there is no use for cleaner fish
        self.cleanEff = cleanEff or 1

        self.surface_ratio_switch = surface_ratio_switch
        if self.surface_ratio_switch:
            #self.biomass = biomass_data #  Whats the point with this?
            self.biomass_update = interp1d(
                x = biomass_data[0], # remember which is which this should be date of fish
                y = biomass_data[1]*0.001, # remember which is which this should be number of fish
                bounds_error = False,
                fill_value = 0
            )
        #  TODO if the controll of this is put into Farm watch out for the done flag
        self.plankton = Planktonic_agent(self.delta_time)
        self.initial_start = initial_start
        self.seasonal_treatment_treashold = seasonal_treatment_treashold
        self.cleaner_death = 0
        self.cleaner_death_ratio = 1

    def update_temp(self):
        self.dayofyear = pd.to_datetime(dates.num2date(self.time+self.initial_start)).dayofyear
        if self.Temp_update_average is None:
            self.temp = self.Temp_update(self.time)
        else:
            self.temp=self.Temp_update_average(self.dayofyear)

    def update(self, attached=0):
        '''
        Hvat skal henda tá ið man uppdaterar Farmina
        :params:
        delta_time      Hvussu nógv skal tíðin flyta seg
        attached        Hvussu nógv lús kemur frá ørðum farms
        '''

        #  If we have been true the last production cycle
        if self.done:
            return None

        self.time += self.delta_time

        self.old_fish_count = self.fish_count
        self.fish_count = self.fish_count_update(self.time)
        #  hvussu nógvur fiskur er deyður relatift
        if self.old_fish_count != 0:
            deth_ratio = min(1, self.fish_count / self.old_fish_count)
        else:
            deth_ratio = 1
        # attchedment_ratio = TODO ger ein attachement ratio  fiskar í byrjandi skulu ikki hava líka nógva lús

        #  TODO This is not in use
        self.update_weigh(self.prod_time)

        #  ratio between surface area between  nógvur fiskur er deyður relatift
        if self.surface_ratio_switch:
            Volumen_farm = (self.biomass_update(self.time)*self.fish_count)/20 #schooling density
            Volumen_grid = 3456000 #160*3*160*3*15
            A_farm  = np.sqrt(Volumen_farm / 15)*np.sqrt(Volumen_farm / 15)
            A_grid = 230400 #160*3*160*3
            surface_grid = np.sqrt(Volumen_grid / 15)#4 * 15 * np.sqrt(Volumen_grid / 15) #+ 160*3*2
            surface_farm = np.sqrt(Volumen_farm / 15)#4 * 15 * np.sqrt(Volumen_farm / 15) #+ (Volumen_farm / 10)
            self.surface_ratio = surface_farm/surface_grid
        else:
            self.surface_ratio = 1

        if self.prod_time < 0:
            self.prod_time += self.delta_time

        elif self.prod_time <= self.prod_len[self.prod_cyc]:

            self.prod_time += self.delta_time

            #============ clenar fish effect========
            if self.CF_data is not None:
                #  if there is no data for cleaner fish dont update it
                if self.use_cleaner_F_update:
                    self.cleaner_fish += self.cleaner_F_update(self.time)
                else:
                    self.cleaner_fish = self.cleaner_count_update(self.time)

                self.cleaner_fish += -self.cleaner_fish*self.delta_time*0.005

                #  How many lice do cleaner fish eat in one delta time step (0.05 per day)
                self.cleaner_death = self.cleaner_fish * self.cleanEff *self.delta_time 

                sum_mobile_lice = np.sum(
                    [
                        np.sum(self.get_fordeiling()[2:6]), 
                        np.sum(self.get_fordeiling()[8:])
                    ]
                )

                if self.cleaner_death <=0 or sum_mobile_lice==0 or np.isnan(self.cleaner_death):
                    self.cleaner_death_ratio = 1
                else:
                    self.cleaner_death_ratio = max([0.001,(1-self.cleaner_death /
                                                           sum_mobile_lice)])


            #================================================#
            #                  _       _       _ _           #
            #  _   _ _ __   __| | __ _| |_ ___| (_) ___ ___  #
            # | | | | '_ \ / _` |/ _` | __/ _ \ | |/ __/ _ \ #
            # | |_| | |_) | (_| | (_| | ||  __/ | | (_|  __/ #
            #  \__,_| .__/ \__,_|\__,_|\__\___|_|_|\___\___| #
            #       |_|                                      #
            #================================================#

            ###########################################
            #       Uppdatera ungar f_lús
            ##########################################
            for stage, lice_list in self.lice_f.items():

                # her reini eg at rokna breiting í bioage fyri hvørja stage bara 1 ferð
                breiting_i_bioage = self.delta_time / ((
                    Lice_agent_f.Hamre_factors_dict[stage] / (
                        Lice_agent_f.b * self.temp ** 2 + 
                        Lice_agent_f.c * self.temp + 
                        Lice_agent_f.d
                    )
                ) * 5)
                deys_fall =  np.exp(- self.lice_mortality_dict[stage] * self.delta_time) *\
                        deth_ratio * (1 if stage in ['Ch1', 'Ch2'] else self.cleaner_death_ratio)
                for lice in lice_list:
                    lice.quik_update(breiting_i_bioage, deys_fall)

            ###########################################
            #       Uppdatera vaksnar f_lús
            ##########################################
            self.adultlice_f.update(self.temp, self.delta_time, deth_ratio,
                                    self.cleaner_death_ratio)

            ###########################################
            #       Skift stadie á ungum f_lús
            ##########################################
            for stage1, stage2, new_key in zip(
                list(self.lice_f.values())[:-1],
                list(self.lice_f.values())[1:],
                list(self.lice_f.keys())[1:]
            ):
                #  Hettar tekur tær elstu fyrst so um nakar hevur skift stadie so er tað tann elsta
                while stage1:
                    stage1[0].update_stage()
                    if stage1[0].get_stage() == new_key:
                        stage2.append(stage1.pop(0))
                    else:
                        break

            ###########################################
            #       Skift stadie á vaksnum f_lús
            ##########################################
            while self.lice_f['Adult']:
                self.lice_f['Adult'][0].update_stage()
                if self.lice_f['Adult'][0].get_stage() == 'Adult_gravid':
                    self.adultlice_f.count += self.lice_f['Adult'].pop(0).count
                else:
                    break  # Hví break her??

            ###########################################
            #       Uppdatera ungar m_lús
            ##########################################
            for stage, lice_list in self.lice_m.items():

                breiting_i_bioage = self.delta_time / ((
                    Lice_agent_m.Hamre_factors_dict[stage] / (
                        Lice_agent_m.b * self.temp ** 2 + 
                        Lice_agent_m.c * self.temp + 
                        Lice_agent_m.d
                    )
                ) * 5)

                deys_fall =  np.exp(- self.lice_mortality_dict[stage] * self.delta_time) *\
                        deth_ratio * (1 if stage in ['Ch1', 'Ch2'] else self.cleaner_death_ratio)
                for lice in lice_list:
                    lice.quik_update(breiting_i_bioage, deys_fall)

            ###########################################
            #       Uppdatera vaksnar m_lús
            ##########################################
            self.adultlice_m.update(self.temp, self.delta_time, deth_ratio,self.cleaner_death_ratio)

            ###########################################
            #       Skift stadie á ungum m_lús
            ##########################################
            for stage1, stage2, new_key in zip(
                list(self.lice_m.values())[:-1],
                list(self.lice_m.values())[1:],
                list(self.lice_m.keys())[1:]
            ):
                #  Hettar tekur teir elstu fyrst so um nakar hevur skift stadie so er tað teir elsta
                while stage1:
                    stage1[0].update_stage()
                    if stage1[0].get_stage() == new_key:
                        stage2.append(stage1.pop(0))
                    else:
                        break

            ###########################################
            #       Skift stadie á vaksnum m_lús
            ##########################################
            while self.lice_m['Pa2']:
                self.lice_m['Pa2'][0].update_stage()
                if self.lice_m['Pa2'][0].get_stage() == 'Adult':
                    self.adultlice_m.count += self.lice_m['Pa2'].pop(0).count
                else:
                    break

            ###########################################
            #  Create a new Lice_agent
            ###########################################
            smitta = (self.L_0 + attached)*self.surface_ratio

            temp_lice_female = Lice_agent_f(self.time, 0, 0,self.lice_mortality)
            mu_f = temp_lice_female.get_mu()
            temp_lice_female.count = smitta / mu_f * (1 - np.exp(-mu_f * self.delta_time))*0.5
            self.lice_f['Ch1'].append(temp_lice_female)

            temp_lice_male = Lice_agent_m(self.time, 0, 0, self.lice_mortality)
            mu_m = temp_lice_male.get_mu()
            temp_lice_male.count = smitta / mu_m * (1 - np.exp(-mu_m * self.delta_time))*0.5
            self.lice_m['Ch1'].append(temp_lice_male)

        elif self.prod_time > self.prod_len[self.prod_cyc]:
            self.fish_count =0
            self.lice_f = {
                'Ch1': [],
                'Ch2': [],
                'Pa1': [],
                'Pa2': [],
                'Adult': [],
            }

            self.lice_m = {
                'Ch1': [],
                'Ch2': [],
                'Pa1': [],
                'Pa2': [],
            }
            self.plankton = Planktonic_agent(self.delta_time)
            self.adultlice_f = Lice_agent_f(self.time, 0, 1000,self.lice_mortality)  # Ja eg skilji ikki orduliga hi tú setur hettar til 1000
            self.adultlice_m = Lice_agent_m(self.time, 0, 1000, self.lice_mortality)
            self.prod_time = -self.fallow[self.prod_cyc]
            self.prod_cyc += 1
            self.cleaner_fish =0
            if self.prod_cyc >= self.prod_len_tjek:
                self.done = True


        #print(self.SliceON)
        if self.NumTreat<=self.num_treat_tjek:
            if self.time > self.treatment[self.NumTreat]: # and self.treatment[self.NumTreat]>0:
                if self.treatment[self.NumTreat]>0:
                    if self.treatment_type[self.NumTreat] in ['FoodTreatment:', 'Emamectin', 'Slice']:
                        self.NumTreat_slice = self.NumTreat
                        self.avlusing('Slice',self.NumTreat_slice, 1, self.temp)
                        self.SliceON = self.treatment_period # length of treatment effect
                    else:
                        self.avlusing('TreatmentY', self.NumTreat, 1, self.temp)

                self.NumTreat += 1

            elif np.isnan(self.treatment[self.NumTreat]):
                self.NumTreat += 1

        if self.SliceON >= 0:
            self.avlusing('Slice', self.NumTreat_slice, 1, self.temp)

            self.SliceON += -self.delta_time

        if self.fish_count!=0:
            if self.seasonal_treatment_treashold:
                relevant_treatment = self.diff_treatment[1][self.diff_treatment[0]==self.dayofyear]
                if np.sum([self.get_fordeiling(calculate=True)[4:6]]) / self.fish_count > relevant_treatment: # OOurt ther sum sigur nær tað er hvat í løbi av árinum
                    self.avlusing('TreatmentX', 1, 1, self.temp)
                    self.num_of_treatments += 1
            else:
                if np.sum([self.get_fordeiling(calculate=True)[4],self.get_fordeiling(calculate=True)[5]]) / self.fish_count > 60:
                    self.avlusing('TreatmentX', 1, 1, self.temp)

        if self.prod_time<0:

            self.get_fordeiling(fallow=1)
        else:

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
            for lice_slag_f in self.lice_f.values():
                for mylice_f in lice_slag_f:
                    mylice_f.TreatmentY(self.treat_eff[:, NumTreat])

            for lice_slag_m in self.lice_m.values():
                for mylice_m in lice_slag_m:
                    mylice_m.TreatmentY(self.treat_eff[:, NumTreat])

            self.adultlice_f.TreatmentY(self.treat_eff[:, NumTreat])
            self.adultlice_m.TreatmentY(self.treat_eff[:, NumTreat])
        # ==========================================
        elif slag == "TreatmentX":
            #TODO ger hettar fyri alt
            for lice_slag_f in self.lice_f.values():
                for mylice_f in lice_slag_f:
                    mylice_f.TreatmentX()

            for lice_slag_m in self.lice_m.values():
                for mylice_m in lice_slag_m:
                    mylice_m.TreatmentX()

            self.adultlice_f.TreatmentX()
            self.adultlice_m.TreatmentX()

        elif slag == "Slice":
            treat_eff = 1-((1-self.treat_eff[:, NumTreat]))*self.delta_time #1-(1-0.95)*1

            for lice_slag_f in self.lice_f.values():
                for mylice_f in lice_slag_f:
                    mylice_f.Slice(treat_eff)

            for lice_slag_m in self.lice_m.values():
                for mylice_m in lice_slag_m:
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
            for  lus in self.lice_f['Ch1']:
                Ch1_f += lus.count
            for  lus in self.lice_f['Ch2']:
                Ch2_f += lus.count
            for  lus in self.lice_f['Pa1']:
                Pa1_f += lus.count
            for  lus in self.lice_f['Pa2']:
                Pa2_f += lus.count
            for  lus in self.lice_f['Adult']:
                Adult_f += lus.count
            Adult_gravid_f += self.adultlice_f.count

            for  lus in self.lice_m['Ch1']:
                Ch1_m += lus.count
            for  lus in self.lice_m['Ch2']:
                Ch2_m += lus.count
            for  lus in self.lice_m['Pa1']:
                Pa1_m += lus.count
            for  lus in self.lice_m['Pa2']:
                Pa2_m += lus.count
            Adult_m += self.adultlice_m.count

            self.__fordeiling__ = [Ch1_f, Ch2_f, Pa1_f, Pa2_f, Adult_f, Adult_gravid_f,
                                   Ch1_m, Ch2_m, Pa1_m, Pa2_m, Adult_m]
        elif fallow == 1:
            Ch1_f, Ch2_f, Pa1_f, Pa2_f, Adult_f,Adult_gravid_f, Ch1_m, Ch2_m, Pa1_m, Pa2_m, Adult_m = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            self.__fordeiling__ = [Ch1_f, Ch2_f, Pa1_f, Pa2_f, Adult_f,Adult_gravid_f,Ch1_m, Ch2_m, Pa1_m, Pa2_m, Adult_m]
        return self.__fordeiling__

    def __repr__(self):
        return self.smitta # 'farmin %s sum hevur %s fiskar' % (self.name, self.fish_count)

    #  TODO hettar skal eftirhiggjast
    def update_weigh(self,prod_time):
        self.W = self.W0*(1-np.exp(-self.K*prod_time))**3  #(1-np.exp(-self.K*(t)))**3 #0.0028 * self.W ** (2 / 3) * (1 - (self.W / 6) ** (1 / 3))
        #(1-np.exp(-self.K*(t)))**3 #0.0028 * self.W ** (2 / 3) * (1 - (self.W / 6) ** (1 / 3))
        return self.W

    def cleaner_F_update(self,time):
        idx = np.where(np.logical_and(self.CF_data[0] > time - 0.001, self.CF_data[0] < time + 0.001))
        if len(idx[0])>0:
            out = np.sum(self.CF_data[1][idx[0][:]])
            return 0 if np.isnan(out) else out
        else:
            return 0

    def insert_treatment(self, treatment_list, treatment_eff):
        assert len(treatment_eff) == len(treatment_list)
        pass

    def insert_prodoction_cycel(self):
        pass
