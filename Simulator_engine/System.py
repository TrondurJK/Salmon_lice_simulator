import numpy as np
import pandas as pd
from matplotlib import dates
from scipy.interpolate import interp1d


class System:
    """
    Class of whole system
    """

    def __init__(self, farms, c_matrix, dayofyear_0=True):
        """
        :params:
        farms               Ein listi av farms sum eru í systeminum
        c_matrix            Ein matrisa sum sigur hvat sambandi er ímillum farms
        time_to_infect      Hvussu leingi tekur tað frá at lísnar verða gjørdar
                                                                til tær seta seg
        """

        self.farms = farms
        self.c_matrix = np.array(c_matrix)
        self.dayofyear_0 = dayofyear_0
        self.initial_start = farms[0].initial_start
        shape = self.c_matrix.shape
        if len(self.c_matrix.shape):
            assert shape[0] == shape[1]
            assert len(self.farms) == shape[0]

        self.pop = [[] for _ in farms]  #  populatión á planktonisku verðinum

    def update(self, time):
        """
        update the whole system
        """

        smittarar = []

        for farm in self.farms:
            farm.update_temp()  # koyrir farm.update_temp fyri at fáa fatur á aktuella temperaturinum og dayof year

            if np.isnan(farm.get_fordeiling()[4]) or farm.fish_count == 0:
                smitta_count = farm.plankton.update(0.0, farm.temp)
            else:
                smitta_count = farm.plankton.update(farm.get_fordeiling()[5], farm.temp)

            smittarar.append(smitta_count)

        if self.dayofyear_0:
            dayofyear = pd.to_datetime(dates.num2date(self.farms[0].time)).dayofyear
        else:
            dayofyear = int(self.farms[0].time - self.initial_start)

        attached_list = []
        for i in range(0, len(self.farms)):
            # print(dayofyear)
            a = self.c_matrix[:, i, dayofyear - 1, :]
            b = np.array(smittarar)

            attached_list.append(np.sum(a * b))

            # frá farm, til farm, dagar síðani sim byrjaði, delay

        for farm, attached in zip(self.farms, attached_list):
            farm.update(attached)
