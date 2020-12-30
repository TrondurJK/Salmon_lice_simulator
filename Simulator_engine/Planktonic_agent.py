import numpy as np
from scipy.interpolate import interp1d
class Planktonic_agent:
    '''
    Hetta kemur at verða tær forskelligu Føstu lísnar
    '''
    __stagees__ = {
        'Napulii':      [1,'Copepodid'],
        'Copepodid':      [np.nan,''],
    }
    def __init__(self, born, delta_time):
        '''
        :params:
        born        Nær er Lúsin fødd
        count       Hvussu nógvar lús eru í hesum chunk [fload]
        bio_age     Hvat er aldurin á lúsini
        '''
        self.born = born
        self.delta_time = delta_time
        #self.age = age
        #self.smittu_count = smittu_count #np.array([0,0])
        self.save_time = 40# how many days we look back
        self.smittu_count = np.zeros((int((self.save_time)/self.delta_time),2))
        self.smittu_count[:,1] = np.linspace((self.save_time-self.delta_time),0,int((self.save_time)/self.delta_time))
        #self.pop = []
        self.interp_egg = interp1d(
            x = [6,12,18], # remember which is which this should be date of fish
            y = [28.9,80.9,90.8], # remember which is which this should be number of fish
            bounds_error = False,
            fill_value = 0
        )

    def update(self,farm,temp):



        alle = (10 * farm.get_fordeiling()[4] * 0.5 / farm.fish_count) / (1 + 10 * farm.get_fordeiling()[4] * 0.5 / farm.fish_count)  # her manglar at deilast við tal av fiski
        #print(temp)
        self.smittu_count[:, 1] += self.delta_time
        self.smittu_count = np.vstack((self.smittu_count, [farm.get_fordeiling()[5] * self.interp_egg(temp) * (alle*0+1)*self.delta_time, 0]))


        #print(self.smittu_count[0,1])
        if self.smittu_count[0,1]>self.save_time:
            #    #self.smittu_count.pop(0)
            self.smittu_count = np.delete(self.smittu_count,1,0)

        #delay_idx  = np.round(delay/self.delta_time) # find how many indicies you have to go back to find time delay
        #delay_idx = delay_idx.astype(int)
        #delay_idx[delay_idx == 0] = 1
        #delay_idx[delay_idx<0]=0
        #delay_idx[delay_idx > self.save_time / self.delta_time]=0
        #print(delay_idx,delay)
        #smittu_count_out = self.smittu_count[-delay_idx,0]
        #if np.sum(smittu_count_out)>0:
        #   hh ='haha'
        #print(smittu_count_out)
        #========== sum all lice in days =============
        self.smittu_count_out =[]
        for i in range(0,40):
            idx = np.where(np.logical_and(np.round(self.smittu_count[:,1], 3)>=i, np.round(self.smittu_count[:,1], 3)<(i+1)))
            self.smittu_count_out.append( np.sum(self.smittu_count[idx,0]) )

        return self.smittu_count_out