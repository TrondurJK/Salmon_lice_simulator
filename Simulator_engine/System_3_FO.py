#from .Planktonic_agent import Planktonic_agent
import numpy as np
import pandas as pd
from matplotlib import dates
from scipy.interpolate import interp1d

class System:
    '''
    Class of whole system
    '''
    def __init__(self, farms, c_matrix,delta_time,temperature_input =None,inital_start=0):
        '''
        :params:
        farms               Ein listi av farms sum eru í systeminum
        c_matrix            Ein matrisa sum sigur hvat sambandi er ímillum farms
        time_to_infect      Hvussu leingi tekur tað frá at lísnar verða gjørdar
                                                                til tær seta seg
        '''
        #  TODO deltatime burda verði ein partur av hesum classanum
        self.delta_time =delta_time
        self.initial_start = inital_start
        self.farms = farms
        self.c_matrix = np.array(c_matrix)
        shape = self.c_matrix.shape
        if len(self.c_matrix.shape):
            assert shape[0] == shape[1]
            assert len(self.farms) == shape[0]
        #self.c_matrix_age = np.array(c_matrix_age)
#        shape = self.c_matrix_age.shape
#        if len(self.c_matrix_age.shape):
#            assert shape[0] == shape[1]
#            assert len(self.farms) == shape[0]


        self.Temp_update = interp1d(
            x = temperature_input[0], # remember which is which this should be date of fish
            y = temperature_input[1], # remember which is which this should be number of fish
            bounds_error = False,
            fill_value = 0
        )
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

        time_doy = pd.to_datetime(dates.num2date(time))
        smittarar = []
        dayofyear = time_doy.dayofyear
        #print(self.farms.name)
        temp = self.Temp_update(time-self.initial_start)
        #print(time,temp)
        #temp = self.Temp_update_average(dayofyear)
        farm_nr = 0
        for farm in self.farms:
            '''
            farm er ein farm og my pop er pop sum hoyrur til somu farm
            '''
            #delay = self.c_matrix[farm_nr, :, dayofyear - 1,:]

            if np.isnan(farm.get_fordeiling()[4]) or farm.fish_count == 0:
                smitta_count = np.zeros((40))
            else:
                smitta_count = farm.plankton.update(farm, temp)

            farm_nr +=1
            smittarar.append(smitta_count)
        #test = np.ones((4, 40))
        attached_list =[]
        for i in range(0,len(self.farms)):
            a = self.c_matrix[:, i, dayofyear - 1, :]
            b = np.array(smittarar)
            #print(a)
            #print(b)
            attached_list.append(np.sum(a*b))

            # frá farm, til farm, dagar síðani sim byrjaði, delay


        #print(attached_list)
        for farm, attached in zip(self.farms, attached_list):
           farm.update(attached)
