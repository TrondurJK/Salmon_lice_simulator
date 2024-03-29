import numpy as np


class Lice_agent:
    def __init__(self, born, count, bio_age, lice_mortality, stage="Ch1"):

        self.born = born
        self.count = count
        self.bio_age = bio_age
        self.__stage__ = stage
        self.lice_mortality = lice_mortality

    def update(self, temp, delta_time, fish_deth_ratio, cleaner_death_ratio):
        """
        Hvat sker við hesi lúsini tá vit veksa timestep við delta_time
        :params:
        temp            Havtempratur
        delta_time      Hvussu nógv skal tíðin breitast
        """

        hampre_factor = self.Hamre_factors_dict[self.__stage__]
        breiting_i_bioage = delta_time / (
            (hampre_factor / (self.b * temp**2 + self.c * temp + self.d)) * 5
        )
        # print(temp,((hampre_factors[0] / (self.b * temp ** 2 + self.c * temp + self.d)) * 5),self.__stage__)
        self.bio_age += breiting_i_bioage

        if np.isin(self.__stage__, ["Ch1", "Ch2"]):
            self.count *= self.deydiliheit(delta_time) * fish_deth_ratio
        else:
            self.count *= (
                self.deydiliheit(delta_time) * fish_deth_ratio * cleaner_death_ratio
            )
        self.update_stage()

    def quik_update(self, breiting_i_bioage, deys_fall):
        self.bio_age += breiting_i_bioage
        self.count *= deys_fall

    def update_stage(self):
        """
        uppdatera stage hjá lúsini
        """
        if self.bio_age > self.__stagees__[self.__stage__][0]:
            self.__stage__ = self.__stagees__[self.__stage__][1]
            return True
        return False

    def get_stage(self):
        """
        fá stage hjá lúsini
        """
        return self.__stage__

    def deydiliheit(self, delta_time, stage=None):
        return np.exp(-self.get_mu(stage) * delta_time)

    def get_mu(self, stage=None):
        """
        Finn útav hvat deyðliheitin á lúsini er
        """
        if stage == None:
            stage = self.get_stage()
        if stage == "Ch1":
            return self.lice_mortality[0]
        elif stage == "Ch2":
            return self.lice_mortality[1]
        elif stage == "Pa1":
            return self.lice_mortality[2]  # 0.025
        elif stage == "Pa2":
            return self.lice_mortality[3]  # 0.025
        elif stage == "Adult":
            return self.lice_mortality[4]  # 0.025
        else:
            return self.lice_mortality[5]  # normal 0.025
        """
        Hendan kodan koyrur tá ið treatmentX verður gjørd
        """

    def treatment(self, treat_eff):
        """
        do a treatment on this louse
        """
        self.count *= treat_eff


class Lice_agent_f(Lice_agent):

    __stagees__ = {
        "Ch1": [1, "Ch2"],
        "Ch2": [2, "Pa1"],
        "Pa1": [3, "Pa2"],
        "Pa2": [4, "Adult"],
        "Adult": [5, "Adult_gravid"],
        "Adult_gravid": [np.nan, ""],
    }

    Hamre_factors_dict = {
        "Ch1": 0.36,
        "Ch2": 0.2,
        "Pa1": 0.2,
        "Pa2": 0.24,
        "Adult": 0.3,  # Should be even longer
        "Adult_gravid": 0.3,
        # 'Adult_egg': [1.3]
    }
    b = 0.000485
    c = 0.008667
    d = 0.003750

    def __init__(self, born, count, bio_age, lice_mortality, **kwargs):

        super().__init__(born, count, bio_age, lice_mortality, **kwargs)


class Lice_agent_m(Lice_agent):

    __stagees__ = {
        "Ch1": [1, "Ch2"],
        "Ch2": [2, "Pa1"],
        "Pa1": [3, "Pa2"],
        "Pa2": [4, "Adult"],
        "Adult": [np.nan, ""],
    }

    Hamre_factors_dict = {
        "Ch1": 0.4,
        "Ch2": 0.2,
        "Pa1": 0.2,
        "Pa2": 0.2,
        "Adult": 0.3,  # Should be even longer
    }
    b = 0.000677
    c = 0.010294
    d = 0.005729

    def __init__(self, born, count, bio_age, lice_mortality, **kwargs):

        super().__init__(born, count, bio_age, lice_mortality, **kwargs)
