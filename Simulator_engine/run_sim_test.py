import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import dates

def init_run_sim(args):
    return run_sim(*args)


def run_sim(delta_time, stop_time, system, inital_start):
    farms = system.farms
    Num_of_treatment_list = [[] for _ in farms]
    stages_f_list = [
        [
            [] for _ in range(6)]
        for _ in farms
    ]
    stages_m_list = [
        [
            [] for _ in range(5)]
        for _ in farms
    ]

    fish_count_save_list  = [[] for _ in farms]
    clean_fish_save_list = [[] for _ in farms]
    attached_save_list = [[] for _ in farms]

    indexis = [[] for _ in farms]

    t = -delta_time
    while t < stop_time:
        t += delta_time  # TODO hettar skal sikkurt flytast inn í nauplii


        system.update(t) #+inital_start)

        for farm, stages_f,stages_m, index, fish_count_save, clean_fish_save, attached_save,Num_of_treatment in \
                zip(farms, stages_f_list, stages_m_list, indexis, fish_count_save_list, clean_fish_save_list,attached_save_list,Num_of_treatment_list):
            Ch1_f, Ch2_f, Pa1_f, Pa2_f, A_f, AG_f,Ch1_m, Ch2_m, Pa1_m, Pa2_m, A_m = [x / max(1, farm.fish_count) for x in farm.get_fordeiling()]

            #print(system.Attached_update(t))

            index.append(farm.time)
            stages_f[0].append(Ch1_f)
            stages_f[1].append(Ch2_f)
            stages_f[2].append(Pa1_f)
            stages_f[3].append(Pa2_f)
            stages_f[4].append(A_f)
            stages_f[5].append(AG_f)
            stages_m[0].append(Ch1_m)
            stages_m[1].append(Ch2_m)
            stages_m[2].append(Pa1_m)
            stages_m[3].append(Pa2_m)
            stages_m[4].append(A_m)

            fish_count_save.append(farm.fish_count)
            clean_fish_save.append(farm.cleaner_fish)

            Num_of_treatment.append(farm.num_of_treatments)

    return index, stages_f_list, stages_m_list, fish_count_save_list,clean_fish_save, farm.name,Num_of_treatment_list


