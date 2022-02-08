from .Lice_agent import Lice_agent_f, Lice_agent_m
from .Planktonic_agent import Planktonic_agent
from .Treatments import Treatments_control

import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from matplotlib import dates

class Farm:
    '''
    Ein klassi til at fylgja við støðuni á teimum einstøku farminum
    '''
    count = 1  #  ein countari til at hjálpa við at geva farminum navn
    treat_id = 1
    def __init__(self, time, delta_time, fish_count = 1_000_000, L_0=0.2, name=None, farm_start=0,
                 prod_len=420_000, fallow=10000, prod_cyc = 0, treatments=None,
                 treatment_type = None, treat_eff=np.array([[]]),
                 weight = 0.2, lusateljingar=[], fish_count_history = None,temperature=None,
                 temperature_Average=None,CF_data =None,biomass_data =None,initial_start=None,
                 cleanEff =None,lice_mortality=None,surface_ratio_switch=False,
                 use_cleaner_F_update=False, seasonal_treatment_treashold=False,
                 treatment_period = None, is_food=None):
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
            treat_eff       Ein matrix av treatment effiicies
            lusateljingar   Ein listi sum sigur nar vit hava lúsateljingar
        '''
        self.time = time


        self.delta_time = delta_time
        self.fish_count_history = fish_count_history
        self.lice_mortality = lice_mortality
        self.fish_count = fish_count

        self.reset_lice()
        self.plankton = Planktonic_agent(self.delta_time)

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

        if isinstance(treatments, Treatments_control):
            self.treat = treatments
        else:
            self.treat = Treatments_control(treatments, treatment_type, treat_eff, treatment_period, is_food=is_food)

        self.num_treat_tjek = np.alen(treatments)-1
        dofy = np.arange(0, 366)
        diff_treat = np.append(dofy[0:int(len(dofy)/2)]*0+20,dofy[int(len(dofy)/2):]*0+80)
        self.diff_treatment = [dofy,diff_treat]
        self.num_of_treatments = 0
        self.prod_len_tjek = len(self.prod_len)
        self.done = False
        self.weight = weight
        self.W0 = 7 # maximum kg av laksinum
        self.K = 0.008  # growth rate í procent pr dag
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
        self.initial_start = initial_start
        self.seasonal_treatment_treashold = seasonal_treatment_treashold
        self.cleaner_death = 0
        self.cleaner_death_ratio = 1
        self.time_to_next_treat = 0

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

        old_fish_count = self.fish_count
        self.fish_count = self.fish_count_update(self.time)
        #  what propotion of the fish is dead
        if old_fish_count != 0:
            deth_ratio = min(1, self.fish_count / old_fish_count)
        else:
            deth_ratio = 1
        # attchedment_ratio = TODO ger ein attachement ratio  fiskar í byrjandi skulu ikki hava líka nógva lús

        #  TODO This is not in use
        self.update_weigh(self.prod_time)

        #  TODO this is only used in update
        #  TODO we shoud calculate this if nessosery
        #  ratio between surface area between  nógvur fiskur er deyður relatift
        if self.surface_ratio_switch:
            self.surface_ratio = self._get_surface_ratio()
        else:
            self.surface_ratio = 1

        #  if we are waiting for the next cycle
        if self.prod_time < 0:
            self.prod_time += self.delta_time

        #  if we are in an active cycle
        elif self.prod_time <= self.prod_len[self.prod_cyc]:

            self.prod_time += self.delta_time

            #============ clenar fish effect========
            if self.CF_data is not None:
                self.updateCF()

            smitta = (self.L_0 + attached)*self.surface_ratio

            #  update young female
            self.update_lice(
                lice_young = self.lice_f,
                Lice_obj = Lice_agent_f,
                adultlice = self.adultlice_f,
                last_young_stage = 'Adult',
                adult_stage = 'Adult_gravid',
                deth_ratio = deth_ratio,
                smitta = .5*smitta
            )
            #  update young male
            self.update_lice(
                lice_young = self.lice_m,
                Lice_obj = Lice_agent_m,
                adultlice = self.adultlice_m,
                last_young_stage = 'Pa2',
                adult_stage = 'Adult',
                deth_ratio = deth_ratio,
                smitta = .5*smitta
            )

        #  if we are at the end of a cycle
        else:
            self.fish_count = 0
            self.reset_lice()

            self.prod_time = -self.fallow[self.prod_cyc]
            self.prod_cyc += 1
            if self.prod_cyc >= self.prod_len_tjek:
                self.done = True

        #  if fish number is zero in a production lice are at the end of a cycle
        if self.fish_count == 0:
            self.reset_lice()

        if self.time_to_next_treat< self.delta_time:
            make_treat = self.treat.apply_Treat(self.time, self.delta_time)
            if make_treat[0]:
                self.avlusing(make_treat[1])
            self.time_to_next_treat = make_treat[2]
        else:
            self.time_to_next_treat -= self.delta_time

        #update the distrubution
        if self.prod_time<0:
            self.get_fordeiling(fallow=1)

        else:
            self.get_fordeiling(calculate=True)

    def avlusing(self, treat_eff):
        '''
        apply a treatment to all the lice
        params : treat_eff is a list (of lenght 6) off big proportion of the lice survive in
                            in the order of [Ch1, Ch2, Pa1, Pa2, Adult, Adult_gravid]
        '''

        for stage, eff in zip(['Ch1', 'Ch2', 'Pa1', 'Pa2', 'Adult'], treat_eff):
            for mylice_f in self.lice_f.get(stage, []):
                mylice_f.treatment(eff)

            for mylice_m in self.lice_m.get(stage, []):
                mylice_m.treatment(eff)

        self.adultlice_f.treatment(treat_eff[5])
        self.adultlice_m.treatment(treat_eff[4])

    def get_fordeiling(self, calculate=False, fallow=0):
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

    def _get_surface_ratio(self):
            Volumen_farm = (self.biomass_update(self.time)*self.fish_count)/20 #schooling density
            Volumen_grid = 3456000 #160*3*160*3*15
            A_farm  = np.sqrt(Volumen_farm / 15)*np.sqrt(Volumen_farm / 15)
            A_grid = 230400 #160*3*160*3
            surface_grid = np.sqrt(Volumen_grid / 15)#4 * 15 * np.sqrt(Volumen_grid / 15) #+ 160*3*2
            surface_farm = np.sqrt(Volumen_farm / 15)#4 * 15 * np.sqrt(Volumen_farm / 15) #+ (Volumen_farm / 10)
            return surface_farm/surface_grid

    def __repr__(self):
        return 'Farm: %s\n nr fish: %s' % (self.name, self.fish_count)

    #  TODO hettar skal eftirhiggjast
    def update_weigh(self,prod_time):
        self.W = self.W0*(1-np.exp(-self.K*prod_time))**3  #(1-np.exp(-self.K*(t)))**3 #0.0028 * self.W ** (2 / 3) * (1 - (self.W / 6) ** (1 / 3))
        #(1-np.exp(-self.K*(t)))**3 #0.0028 * self.W ** (2 / 3) * (1 - (self.W / 6) ** (1 / 3))
        return self.W

    def updateCF(self):
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

    def cleaner_F_update(self,time):
        idx = np.where(np.logical_and(self.CF_data[0] > time - 0.001, self.CF_data[0] < time + 0.001))
        if len(idx[0])>0:
            out = np.sum(self.CF_data[1][idx[0][:]])
            return 0 if np.isnan(out) else out
        else:
            return 0

    def update_lice(
        self,
        lice_young,
        Lice_obj,
        adultlice,
        last_young_stage,
        adult_stage,
        deth_ratio,
        smitta
    ):
        ###########################################
        #       update young lice
        ##########################################
        for stage, lice_list in lice_young.items():

            # her reini eg at rokna breiting í bioage fyri hvørja stage bara 1 ferð
            breiting_i_bioage = self.delta_time / ((
                Lice_obj.Hamre_factors_dict[stage] / (
                Lice_obj.b * self.temp ** 2 + 
                Lice_obj.c * self.temp + 
                Lice_obj.d
                )
            ) * 5)

            deys_fall =  np.exp(- self.lice_mortality_dict[stage] * self.delta_time) *\
                    deth_ratio * (1 if stage in ['Ch1', 'Ch2'] else self.cleaner_death_ratio)
            for lice in lice_list:
                lice.quik_update(breiting_i_bioage, deys_fall)

        ###########################################
        #       update adult lice
        ##########################################
        adultlice.update(self.temp, self.delta_time, deth_ratio,
                                self.cleaner_death_ratio)

        ###########################################
        #       update the stage of the lice
        ##########################################
        for stage1, stage2, new_key in zip(
            list(lice_young.values())[:-1],
            list(lice_young.values())[1:],
            list(lice_young.keys())[1:]
        ):
            #  update the stage of a single stage from the oldest until we have found them all
            while stage1:
                stage1[0].update_stage()
                if stage1[0].get_stage() == new_key:
                    stage2.append(stage1.pop(0))
                else:
                    break

        ###########################################
        #       Change stage of the oldest younglings 
        ##########################################
        while lice_young[last_young_stage]:
            lice_young[last_young_stage][0].update_stage()
            if lice_young[last_young_stage][0].get_stage() == adult_stage:
                adultlice.count += lice_young[last_young_stage].pop(0).count
            else:
                break

        ###########################################
        #  Create a new Lice_agent
        ###########################################

        temp_lice = Lice_obj(self.time, 0, 0, self.lice_mortality)
        mu = temp_lice.get_mu()
        temp_lice.count = (smitta / mu) * (1 - np.exp(-mu * self.delta_time))
        lice_young['Ch1'].append(temp_lice)

    def reset_lice(self):
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
        #  make the old lice objects 
        self.adultlice_f = Lice_agent_f(self.time, 0, 1000,self.lice_mortality, stage='Adult_gravid')
        self.adultlice_m = Lice_agent_m(self.time, 0, 1000, self.lice_mortality, stage='Adult')

    def insert_treatment(self, treatment_list, treatment_eff):
        assert len(treatment_eff) == len(treatment_list)
        pass

    def insert_prodoction_cycel(self):
        pass
