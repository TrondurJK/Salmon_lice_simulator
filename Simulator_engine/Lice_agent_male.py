import numpy as np


class Lice_agent_m:
    '''
    Hetta kemur at verða tær forskelligu Føstu lísnar
    '''
    __stagees__ = {
        'Ch1': [1, 'Ch2'],
        'Ch2': [2, 'Pa1'],
        'Pa1': [3, 'Pa2'],
        'Pa2': [4, 'Adult'],
        'Adult': [np.nan, ''],
    }

    def __init__(self, born, count, bio_age,lice_mortality):
        '''
        :params:
        born        Nær er Lúsin fødd
        count       Hvussu nógvar lús eru í hesum chunk [fload]
        bio_age     Hvat er aldurin á lúsini
        '''
        self.born = born
        self.count = count
        self.bio_age = bio_age
        self.__stage__ = 'Ch1'
        self.lice_mortality = lice_mortality

        self.Hamre_factors_dict = {
            'Ch1': [0.4],
            'Ch2': [0.2],
            'Pa1': [0.2],
            'Pa2': [0.2],
            'Adult': [0.3], # Should be even longer
            #'Adult_gravid': [0.3],
            # 'Adult_egg': [1.3]

        }
        self.b = 0.000677
        self.c = 0.010294
        self.d = 0.005729

    def update(self, temp, delta_time, fish_deth_ratio, cleaner_death_ratio):
        '''
        Hvat sker við hesi lúsini tá vit veksa timestep við delta_time
        :params:
        temp            Havtempratur
        delta_time      Hvussu nógv skal tíðin breitast
        '''

        hampre_factors = self.Hamre_factors_dict[self.__stage__]
        breiting_i_bioage = delta_time / ((hampre_factors[0] / (self.b * temp ** 2 + self.c * temp + self.d)) * 5)
        #print(temp,((hampre_factors[0] / (self.b * temp ** 2 + self.c * temp + self.d)) * 5),self.__stage__)
        self.bio_age += breiting_i_bioage

        if np.isin(self.__stage__,['Ch1','Ch2']):
            self.count *= self.deydiliheit(delta_time) * fish_deth_ratio
        else:
            self.count *= self.deydiliheit(delta_time) * fish_deth_ratio * cleaner_death_ratio
        self.update_stage()

    def update_stage(self):
        '''
        uppdatera stage hjá lúsini
        '''
        if self.bio_age > self.__stagees__[self.__stage__][0]:
            self.__stage__ = self.__stagees__[self.__stage__][1]
            return True
        return False

    def get_stage(self):
        '''
        fá stage hjá lúsini
        '''
        return self.__stage__

    def deydiliheit(self, delta_time, stage=None):
        return np.exp(- self.get_mu(stage) * delta_time)

    def get_mu(self, stage=None):
        '''
        Finn útav hvat deyðliheitin á lúsini er
        '''
        if stage == None:
            stage = self.get_stage()
        if stage == 'Ch1':
            return self.lice_mortality[0]
        elif stage == 'Ch2':
            return self.lice_mortality[1]
        elif stage == 'Pa1':
            return self.lice_mortality[2]  # 0.025
        elif stage == 'Pa2':
            return self.lice_mortality[3]  # 0.025
        elif stage == 'Adult':
            return self.lice_mortality[4]  # 0.025
        else:
            return self.lice_mortality[5]  # normal 0.025
        '''
        Hendan kodan koyrur tá ið treatmentX verður gjørd
        '''

    def TreatmentY(self, treat_eff):
        if self.get_stage() in ['Ch1']:
            self.count = self.count * treat_eff[0]
        if self.get_stage() in ['Ch2']:
            self.count = self.count * treat_eff[1]
        if self.get_stage() in ['Pa1']:
            self.count = self.count * treat_eff[2]
        if self.get_stage() in ['Pa2']:
            self.count = self.count * treat_eff[3]
        if self.get_stage() in ['Adult']:
            self.count = self.count * treat_eff[4]
        if self.get_stage() in ['Adult_gravid']:
            self.count = self.count * treat_eff[5]

    def TreatmentX(self):
        if self.get_stage() in ['Ch1', 'Ch2', 'Pa1', 'Pa2', 'Adult']:
            self.count = self.count * 0.20

    def Slice(self, treat_eff):
        if self.get_stage() in ['Ch1']:
            self.count = self.count * treat_eff[0]
        if self.get_stage() in ['Ch2']:
            self.count = self.count * treat_eff[1]
        if self.get_stage() in ['Pa1']:
            self.count = self.count * treat_eff[2]
        if self.get_stage() in ['Pa2']:
            self.count = self.count * treat_eff[3]
        if self.get_stage() in ['Adult']:
            self.count = self.count * treat_eff[4]
        if self.get_stage() in ['Adult_gravid']:
            self.count = self.count * treat_eff[5]

    def H2O2(self):
        if self.get_stage() in ['Pa1', 'Pa2', 'Adult']:
            self.count = self.count * 0.3
        if self.get_stage() in ['Ch1', 'Ch2']:
            self.count = self.count * 0.80

    def Salmosan(self):
        if self.get_stage() in ['Ch1', 'Ch2', 'Pa1', 'Pa2', 'Adult']:
            self.count = self.count * 0.85

    def Azametiphos(self):
        if self.get_stage() in ['Ch1', 'Ch2', 'Pa1', 'Pa2', 'Adult']:
            self.count = self.count * 0.45

    def Diflubenzuron(self):
        if self.get_stage() in ['Ch1', 'Ch2', 'Pa1', 'Pa2', 'Adult']:
            self.count = self.count * 0.95

    def Pyretroid(self):
        if self.get_stage() in ['Ch1', 'Ch2', 'Pa1', 'Pa2', 'Adult']:
            self.count = self.count * 0.80

    def Pyretroid_Azametiphos(self):
        if self.get_stage() in ['Ch1', 'Ch2', 'Pa1', 'Pa2', 'Adult']:
            self.count = self.count * 0.80

    def FV(self):
        if self.get_stage() in ['Ch1', 'Ch2', 'Pa1', 'Pa2', 'Adult']:
            self.count = self.count * 0.60

    def Optilicer(self):
        if self.get_stage() in ['Ch1', 'Ch2', 'Pa1', 'Pa2', 'Adult']:
            self.count = self.count * 0.45
