import numpy as np
import pandas as pd
from matplotlib import dates
from scipy.interpolate import interp1d

class System:
    '''
    Class of whole system
    '''
    def __init__(self, farms, c_matrix):
        '''
        :params:
        farms               Ein listi av farms sum eru í systeminum
        c_matrix            Ein matrisa sum sigur hvat sambandi er ímillum farms
        time_to_infect      Hvussu leingi tekur tað frá at lísnar verða gjørdar
                                                                til tær seta seg
        '''

        self.farms = farms
        self.c_matrix = np.array(c_matrix)
        shape = self.c_matrix.shape
        if len(self.c_matrix.shape):
            assert shape[0] == shape[1]
            assert len(self.farms) == shape[0]

        self.pop = [[] for _ in farms] #  populatión á planktonisku verðinum

    def update(self, time):
        '''
        uppdatera alt systemi við 'delta_time' har havtempraturin er 'temp'
        Hettar:
            uppdater aldurin á øllum lísnum
            ger níggjar lús útfrá smittutrístinum
            og uppdaterar smittutrísti
        :params:
        temp            Havtempraturur
        delta_time      breiting í tíð
        '''

        smittarar = []

        farm_nr = 0
        for farm in self.farms:
            '''
            farm er ein farm og my pop er pop sum hoyrur til somu farm
            '''

            farm.update_temp() # koyrir farm.update_temp fyri at fáa fatur á aktuella temperaturinum og dayof year
            temp = farm.temp
            dayofyear = farm.dayofyear

            if np.isnan(farm.get_fordeiling()[4]) or farm.fish_count == 0:
                smitta_count = np.zeros((40))
            else:
                smitta_count = farm.plankton.update(farm, temp)

            farm_nr +=1
            smittarar.append(smitta_count)

        attached_list =[]
        for i in range(0,len(self.farms)):
            a = self.c_matrix[:, i, dayofyear - 1, :]
            b = np.array(smittarar)

            attached_list.append(np.sum(a*b))

            # frá farm, til farm, dagar síðani sim byrjaði, delay


        for farm, attached in zip(self.farms, attached_list):
           farm.update(attached)
