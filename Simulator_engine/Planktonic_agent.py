import numpy as np
from scipy.interpolate import interp1d
class Planktonic_agent:
    '''
    Hetta kemur at verða tær forskelligu Føstu lísnar
    '''

    def __init__(self, delta_time, save_time=40):
        '''
        :params:
        born        Nær er Lúsin fødd # TODO Fjerna meg
        save_time   how many days we look back
        '''

        self.delta_time = delta_time

        cols = int(np.ceil(save_time/delta_time))

        # the vector that holds the data for the last {save_time} steps
        self.smittu_count = np.zeros(
            (cols),
            dtype=float
        )

        #  The matrix to convert from raw data to lice per day
        self.A = np.zeros((save_time, cols), dtype=float)

        for i in range(cols):
            start = i*delta_time
            start_iter = start
            end = (i+1)*delta_time
            while start_iter < min(end, save_time):
                c = np.floor(start_iter) + 1
                if c < end:
                    self.A[int(np.floor(start_iter)), i] = (c - start_iter)/(end - start)
                    start_iter = c
                else:
                    self.A[int(np.floor(start_iter)), i] = (end - start_iter)/(end - start)
                    start_iter = end

        self.interp_egg = interp1d(
            x = [6,12,18], # remember which is which this should be date of fish
            y = [28.9,80.9,90.8], # remember which is which this should be number of fish
            bounds_error = False,
            fill_value = 0
        )

    def update(self, farm, temp):

        #alle = (10 * farm.get_fordeiling()[4] * 0.5 / farm.fish_count) / (1 + 10 * farm.get_fordeiling()[4] * 0.5 / farm.fish_count)  # her manglar at deilast við tal av fiski
        #self.smittu_count = np.vstack((self.smittu_count, [farm.get_fordeiling()[5] * self.interp_egg(temp) * (alle*0+1)*self.delta_time, 0]))
        self.smittu_count = np.hstack(([farm.get_fordeiling()[5] * self.interp_egg(temp) * self.delta_time], self.smittu_count[:-1]))

        smittu_count_out_test = np.dot(self.A, self.smittu_count)
        return smittu_count_out_test
