from numbers import Number

from .Lice_agent import Lice_agent_f, Lice_agent_m
from .Planktonic_agent import Planktonic_agent
from .Treatments import Treatments_control

import numpy as np
from scipy.interpolate import interp1d


class Farm:
    """
    Ein klassi til at fylgja við støðuni á teimum einstøku farminum
    """

    count = 1  #  ein countari til at hjálpa við at geva farminum navn
    treat_id = 1

    def __init__(
        self,
        delta_time=0.5,
        fish_count=200_000,
        L_0=2000,
        name=None,
        initial_start=0,
        farm_start=0,
        prod_len=420,
        fallow=[90],
        treatments=None,
        treatment_type=None,
        treatment_period=None,
        treatment_is_food=None,
        treat_eff=np.array([[]]),
        treat_automatic_eff=None,
        treat_automatic_thres=None,
        fish_count_history=None,
        temperature=None,
        mean_temprature=10,
        CF_data=None,
        cleanEff=None,
        cleanMean=0.2,
        cleanEff_method="Interp",
        CF_lice_min=0.0,
        lice_mortality=[0.01, 0.01, 0.02, 0.02, 0.02, 0.02],
        surface_ratio_switch=False,
        surface_ratio_k=0.15,
        biomass_data=None,
    ):
        """
        :params:
            delta_time             Time step
            fish_count             if fish_count_history is not spesified, the constant fish_count
            L_0                    External infection (total amount of lice per day)
            name                   name of the farm
            initial_start          start timestep of the simplation (days)
            farm_start             delay between initial_start and the start of the farm
            prod_len               how long is the production (days)
            fallow                 lenght of the fallowing period (days)
            treatments             a list of treatments dates (days after initial_start)
            treatment_type         if defined List[str] of same lenght as treatments
            treatment_period       how long does the effect of the treatment last, if defined List[int] of same lenght as treatments
            treatment_is_food      an list that specefies if the treatments are food based, if defined List[int] of same lenght as treatments
            treat_eff              effientsy of the treatments on the 6 stages of lice (np.array of shape (6, len(treatments)))
            treat_automatic_thres  Initate a automatic treatment threshold and define level (lice/fish)
            treat_automatic_eff    Define treatment effeciency in the automatic treatment
            fish_count_history     records of fish count ([List[int], List[int]], the first list is days after initial_start the sekend is fish_count)
            temprature             records of temprature same structure as fish_count_history, if date is outsite of the range of the recurds it will be set to mean_temprature
            mean_temprature        the standard value of the temprature if it is not set
            CF_data                records of Cleaner fish same structure as fish_count_history
            cleanEff               the effinsy of the cleaner fish
            lice_mortality         mortalety of the 6 stages of lice (no diffrense between male and female)
            surface_ratio_switch   boolian if we shall use surface_ratio for lice attacment (Experimental feature): 0 = not used, 1 = constant receiving, 2 = functional response II, 3 = dependant on bio mass
            surface_ratio_k        A konstant for funtional respones type II surface ratio area
            biomass_data           records of biomass fish same structure as fish_count_history (Experimental feature)
            CF_lice_min            the minimum level of moblice lice in order for clenar fish to work

        """
        #  TODO figure out temprature
        #  TODO follow self.time
        self.time = initial_start

        self.delta_time = delta_time
        self.lice_mortality = lice_mortality

        self.reset_lice()
        self.plankton = Planktonic_agent(self.delta_time)

        if len(lice_mortality) >= 6:
            pass
        else:
            raise Exception("lice_mortality skal minst verða 6 langur")
        self.lice_mortality_dict = {
            slag: mortality
            for slag, mortality in zip(
                ["Ch1", "Ch2", "Pa1", "Pa2", "Adult", "Adult_gravid"], lice_mortality
            )
        }
        if isinstance(L_0, Number):
            self.L_0 = lambda x: L_0
        else:
            self.L_0 = interp1d(
                x=L_0[0], y=L_0[1], bounds_error=False, fill_value=0  #  #
            )

        # Hvissi einki navn er sett so gerða vit eitt
        if name == None:
            self.name = "farm_%s" % self.__class__.count
            self.__class__.count += 1
        else:
            self.name = name
        self.prod_len = prod_len

        #  make it so that if fallow is a number the code does not break
        self.fallow = fallow
        self.prod_time = -farm_start
        self.__fordeiling__ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.prod_cyc = 0
        self.treatment = treatments

        if isinstance(treatments, Treatments_control):
            self.treat = treatments
        else:
            self.treat = Treatments_control(
                treatments,
                treatment_type,
                treat_eff,
                treatment_period,
                is_food=treatment_is_food,
            )

        self.num_treat_tjek = len(treatments) - 1

        self.treat_automatic_time = []
        if treat_automatic_thres is not None:
            if isinstance(treat_automatic_thres, Number):
                self.treat_automatic_thres = lambda x: treat_automatic_thres

            else:
                self.treat_automatic_thres = interp1d(
                    x=treat_automatic_thres[0],  #
                    y=treat_automatic_thres[1],  #
                    bounds_error=False,
                    fill_value=0,
                )
            self.treat_automatic_eff = treat_automatic_eff
        else:
            self.treat_automatic_thres = lambda x: np.inf
            self.treat_automatic_eff = [1, 1, 1, 1, 1, 1]

        dofy = np.arange(0, 366)
        diff_treat = np.append(
            dofy[0 : int(len(dofy) / 2)] * 0 + 20, dofy[int(len(dofy) / 2) :] * 0 + 80
        )
        self.diff_treatment = [dofy, diff_treat]
        self.prod_len_tjek = len(self.prod_len)
        self.done = False
        self.W0 = 7  # maximum kg av laksinum
        self.K = 0.008  # growth rate í procent pr dag

        self.fish_count = fish_count
        if fish_count_history:
            self.fish_count_update = interp1d(
                x=fish_count_history[
                    0
                ],  # remember which is which this should be date of fish
                y=fish_count_history[
                    1
                ],  # remember which is which this should be number of fish
                bounds_error=False,
                fill_value=0,
            )
            self.fish_count = self.fish_count_update(self.time)
        else:
            self.fish_count_update = lambda x: fish_count

        if temperature:
            self.Temp_update = interp1d(
                x=temperature[0],  # Date
                y=temperature[1],  # Temperature
                bounds_error=False,
                fill_value=mean_temprature,
            )
        else:
            self.Temp_update = lambda x: mean_temprature
        self.update_temp()  # Update to newest temperature

        if CF_data is not None:
            self.cleaner_count_update = interp1d(
                x=CF_data[0],  # remember which is which this should be date of fish
                y=CF_data[1],  # remember which is which this should be number of fish
                bounds_error=False,
                fill_value=0,
            )
        else:
            self.cleaner_count_update = lambda x: 0
        self.CF_data = CF_data

        self.cleaner_fish = 0
        #  TODO what to do with cleanEff if there is no use for cleaner fish

        self.cleanMean = cleanMean
        self.cleanEff = cleanEff
        self.cleanEff_method = cleanEff_method
        self._make_CleanEff_update()
        self.cleanEff_update()

        self.surface_ratio_switch = surface_ratio_switch
        if self.surface_ratio_switch:
            # self.biomass = biomass_data #  Whats the point with this?
            self.biomass_update = interp1d(
                x=biomass_data[
                    0
                ],  # remember which is which this should be date of fish
                y=biomass_data[1]
                * 0.001,  # remember which is which this should be number of fish
                bounds_error=False,
                fill_value=0,
            )
            self.max_fish_biomass = np.max(biomass_data[1] * 0.001)
        self.surface_ratio_k = surface_ratio_k
        #  TODO if the controll of this is put into Farm watch out for the done flag
        self.initial_start = initial_start
        self.cleaner_death = 0
        self.cleaner_death_ratio = 1
        self.CF_lice_min = (
            CF_lice_min  # the minimum amount moblie lice for cleaner fish to mork
        )
        self.time_to_next_treat = 0

        self.treat_counter = 0

    def _make_CleanEff_update(self):
        if (
            self.cleanEff
            and self.cleanEff != [[], []]
        ):

            if isinstance(self.cleanEff, list):

                if self.cleanEff_method == "previous":
                    cleanMean = (self.cleanMean, self.cleanEff[1][-1])
                else:
                    cleanMean = self.cleanMean

                self.CleanEff_update = interp1d(
                    x=self.cleanEff[0],  # Date
                    y=self.cleanEff[1],  # Cleanerfish effeciency
                    kind=self.cleanEff_method,
                    bounds_error=False,
                    fill_value=cleanMean,
                )
            else:
                self.CleanEff_update = lambda x: self.cleanEff
        else:
            self.CleanEff_update = lambda x: self.cleanMean

    def update_temp(self):
        self.temp = self.Temp_update(self.time)

    def cleanEff_update(self):
        self.cf_eff = self.CleanEff_update(self.time)

    def update(self, attached=0):
        """
        Hvat skal henda tá ið man uppdaterar Farmina
        :params:
        delta_time      Hvussu nógv skal tíðin flyta seg
        attached        Hvussu nógv lús kemur frá ørðum farms
        """

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
        if self.surface_ratio_switch == 3:
            self.surface_ratio = self._get_surface_ratio()
        elif self.surface_ratio_switch == 2:
            x_max = (
                self.biomass_update(self.time) * self.fish_count
            ) / self.max_fish_biomass
            a = 100
            FR_typeII = (x_max * a) / (1 + x_max * a) + 0.01
            self.surface_ratio = self.surface_ratio_k * FR_typeII
        elif self.surface_ratio_switch == 1:
            self.surface_ratio = self.surface_ratio_k
        else:
            self.surface_ratio = 1

        #  if we are waiting for the next cycle
        if self.prod_time < 0:
            self.prod_time += self.delta_time

        #  if we are in an active cycle
        elif self.prod_time <= self.prod_len[self.prod_cyc]:

            self.prod_time += self.delta_time

            # ============ clenar fish effect========
            if self.CF_data is not None:
                self.updateCF()
                self.cleanEff_update()

            smitta = (self.L_0(self.time) + attached) * self.surface_ratio

            #  update young female
            self.update_lice(
                lice_young=self.lice_f,
                Lice_obj=Lice_agent_f,
                adultlice=self.adultlice_f,
                last_young_stage="Adult",
                adult_stage="Adult_gravid",
                deth_ratio=deth_ratio,
                smitta=0.5 * smitta,
            )
            #  update young male
            self.update_lice(
                lice_young=self.lice_m,
                Lice_obj=Lice_agent_m,
                adultlice=self.adultlice_m,
                last_young_stage="Pa2",
                adult_stage="Adult",
                deth_ratio=deth_ratio,
                smitta=0.5 * smitta,
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

        if self.treat_automatic_thres:

            if self.fish_count and np.sum(
                self.get_fordeiling()[4:6]
            ) / self.fish_count > self.treat_automatic_thres(self.time):

                self.treatments(self.treat_automatic_eff)
                self.treat_automatic_time.append(self.time)
                self.treat_counter += 1

        if self.time_to_next_treat < self.delta_time:
            make_treat = self.treat.apply_Treat(self.time, self.delta_time)
            if make_treat[0]:
                self.treatments(make_treat[1])
            self.time_to_next_treat = make_treat[2]
        else:
            self.time_to_next_treat -= self.delta_time

        # update the distrubution
        if self.prod_time < 0:
            self.get_fordeiling(fallow=1)

        else:
            self.get_fordeiling(calculate=True)

    def treatments(self, treat_eff):
        """
        apply a treatment to all the lice
        params : treat_eff is a list (of lenght 6) off big proportion of the lice survive in
                            in the order of [Ch1, Ch2, Pa1, Pa2, Adult, Adult_gravid]
        """

        for stage, eff in zip(["Ch1", "Ch2", "Pa1", "Pa2", "Adult"], treat_eff):
            for mylice_f in self.lice_f.get(stage, []):
                mylice_f.treatment(eff)

            for mylice_m in self.lice_m.get(stage, []):
                mylice_m.treatment(eff)

        self.adultlice_f.treatment(treat_eff[5])
        self.adultlice_m.treatment(treat_eff[4])

    def get_fordeiling(self, calculate=False, fallow=0):
        """
        Gevur ein lista av teimum forskelligu stages
        :params:
        calculate       skal man brúka tíð sístu frodeilingina ella skal hettar roknast
        """

        if calculate:
            (
                Ch1_f,
                Ch2_f,
                Pa1_f,
                Pa2_f,
                Adult_f,
                Adult_gravid_f,
                Ch1_m,
                Ch2_m,
                Pa1_m,
                Pa2_m,
                Adult_m,
            ) = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            # Chalimus 1, Chalimus 2, Pre-Adult 1, Pre-Adult 2, Adult
            for lus in self.lice_f["Ch1"]:
                Ch1_f += lus.count
            for lus in self.lice_f["Ch2"]:
                Ch2_f += lus.count
            for lus in self.lice_f["Pa1"]:
                Pa1_f += lus.count
            for lus in self.lice_f["Pa2"]:
                Pa2_f += lus.count
            for lus in self.lice_f["Adult"]:
                Adult_f += lus.count
            Adult_gravid_f += self.adultlice_f.count

            for lus in self.lice_m["Ch1"]:
                Ch1_m += lus.count
            for lus in self.lice_m["Ch2"]:
                Ch2_m += lus.count
            for lus in self.lice_m["Pa1"]:
                Pa1_m += lus.count
            for lus in self.lice_m["Pa2"]:
                Pa2_m += lus.count
            Adult_m += self.adultlice_m.count

            self.__fordeiling__ = [
                Ch1_f,
                Ch2_f,
                Pa1_f,
                Pa2_f,
                Adult_f,
                Adult_gravid_f,
                Ch1_m,
                Ch2_m,
                Pa1_m,
                Pa2_m,
                Adult_m,
            ]
        elif fallow == 1:
            (
                Ch1_f,
                Ch2_f,
                Pa1_f,
                Pa2_f,
                Adult_f,
                Adult_gravid_f,
                Ch1_m,
                Ch2_m,
                Pa1_m,
                Pa2_m,
                Adult_m,
            ) = (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )
            self.__fordeiling__ = [
                Ch1_f,
                Ch2_f,
                Pa1_f,
                Pa2_f,
                Adult_f,
                Adult_gravid_f,
                Ch1_m,
                Ch2_m,
                Pa1_m,
                Pa2_m,
                Adult_m,
            ]
        return self.__fordeiling__

    def _get_surface_ratio(self):
        Volumen_farm = (
            self.biomass_update(self.time) * self.fish_count
        ) / 20  # schooling density
        Volumen_grid = 3456000  # 160*3*160*3*15
        A_farm = np.sqrt(Volumen_farm / 15) * np.sqrt(Volumen_farm / 15)
        A_grid = 230400  # 160*3*160*3
        surface_grid = np.sqrt(Volumen_grid / 15)
        surface_farm = np.sqrt(Volumen_farm / 15)
        return surface_farm / surface_grid

    def __repr__(self):
        return "Farm: %s\n nr fish: %s" % (self.name, self.fish_count)

    #  TODO hettar skal eftirhiggjast
    def update_weigh(self, prod_time):
        self.W = (
            self.W0 * (1 - np.exp(-self.K * prod_time)) ** 3
        )  # (1-np.exp(-self.K*(t)))**3 #0.0028 * self.W ** (2 / 3) * (1 - (self.W / 6) ** (1 / 3))
        # (1-np.exp(-self.K*(t)))**3 #0.0028 * self.W ** (2 / 3) * (1 - (self.W / 6) ** (1 / 3))
        return self.W

    def updateCF(self):
        #  if there is no data for cleaner fish don't update it
        if self.fish_count == 0:
            PAA_lice = 0
        else:
            PAA_lice = np.sum(self.get_fordeiling()[4:6]) / self.fish_count
        self.cleaner_fish = self.cleaner_count_update(self.time)
        self.cleanEff_update()
        #  How many lice do cleaner fish eat in one delta time step (0.05 per day)
        self.cleaner_death = self.cleaner_fish * self.cf_eff * self.delta_time

        sum_mobile_lice = np.sum(
            [np.sum(self.get_fordeiling()[2:6]), np.sum(self.get_fordeiling()[8:])]
        )
        if self.cleaner_death <= 0:
            self.cleaner_death_ratio = 1

        elif np.isnan(self.cleaner_death):
            self.cleaner_death_ratio = 1

        elif PAA_lice <= self.CF_lice_min:
            self.cleaner_death_ratio = 1
        elif PAA_lice > self.CF_lice_min:
            not_below_CF_min = self.CF_lice_min / (
                PAA_lice
            )

            self.cleaner_death_ratio = max(
                [not_below_CF_min, (1 - self.cleaner_death / sum_mobile_lice)]
            )

    def update_lice(
        self,
        lice_young,
        Lice_obj,
        adultlice,
        last_young_stage,
        adult_stage,
        deth_ratio,
        smitta,
    ):
        ###########################################
        #       update young lice
        ##########################################
        for stage, lice_list in lice_young.items():

            # her reini eg at rokna breiting í bioage fyri hvørja stage bara 1 ferð
            breiting_i_bioage = self.delta_time / (
                (
                    Lice_obj.Hamre_factors_dict[stage]
                    / (
                        Lice_obj.b * self.temp**2
                        + Lice_obj.c * self.temp
                        + Lice_obj.d
                    )
                )
                * 5
            )

            deys_fall = (
                np.exp(-self.lice_mortality_dict[stage] * self.delta_time)
                * deth_ratio
                * (1 if stage in ["Ch1", "Ch2"] else self.cleaner_death_ratio)
            )
            for lice in lice_list:
                lice.quik_update(breiting_i_bioage, deys_fall)

        ###########################################
        #       update adult lice
        ##########################################
        adultlice.update(
            self.temp, self.delta_time, deth_ratio, self.cleaner_death_ratio
        )

        ###########################################
        #       update the stage of the lice
        ##########################################
        for stage1, stage2, new_key in zip(
            list(lice_young.values())[:-1],
            list(lice_young.values())[1:],
            list(lice_young.keys())[1:],
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
        lice_young["Ch1"].append(temp_lice)

    def reset_lice(self):
        self.lice_f = {
            "Ch1": [],
            "Ch2": [],
            "Pa1": [],
            "Pa2": [],
            "Adult": [],
        }

        self.lice_m = {
            "Ch1": [],
            "Ch2": [],
            "Pa1": [],
            "Pa2": [],
        }
        #  make the old lice objects
        self.adultlice_f = Lice_agent_f(
            self.time, 0, 1000, self.lice_mortality, stage="Adult_gravid"
        )
        self.adultlice_m = Lice_agent_m(
            self.time, 0, 1000, self.lice_mortality, stage="Adult"
        )

    def insert_treatment(self, treatment_list, treatment_eff):
        assert len(treatment_eff) == len(treatment_list)
        pass

    def insert_prodoction_cycel(self):
        pass
