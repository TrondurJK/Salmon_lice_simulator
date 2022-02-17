import pandas as pd
import numpy as np

class Sim:
    pass


def init_run_sim(args):
    return run_sim(*args)


def run_sim(delta_time,inital_start, stop_time, system):
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
    temp_save_list = [[] for _ in farms]
    clean_fish_save_list = [[] for _ in farms]
    attached_save_list = [[] for _ in farms]

    indexis = [[] for _ in farms]
    unique_id = []
    t = inital_start-delta_time
    sim = Sim()
    while t < stop_time:
        t += delta_time  # TODO hettar skal sikkurt flytast inn í nauplii

        system.update(t)

        for id,(farm, stages_f,stages_m, index, fish_count_save,temp_save, clean_fish_save, attached_save,Num_of_treatment) in \
                enumerate(zip(farms, stages_f_list, stages_m_list, indexis, fish_count_save_list,temp_save_list, clean_fish_save_list,attached_save_list,Num_of_treatment_list)):
            Ch1_f, Ch2_f, Pa1_f, Pa2_f, A_f, AG_f,Ch1_m, Ch2_m, Pa1_m, Pa2_m, A_m = [x / max(1, farm.fish_count) for x in farm.get_fordeiling()]

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
            temp_save.append(farm.temp)
            clean_fish_save.append(farm.cleaner_fish)

            Num_of_treatment.append(farm.num_of_treatments)
            unique_id.append(id)

    for id in np.unique(unique_id):
        d = {
                "Date":index,
                "Nfish": fish_count_save_list[id],
                "AF_gravid": stages_f_list[id][5],
                "AF": stages_f_list[id][4],
                "PA_2_f": stages_f_list[id][3],
                "PA_1_f": stages_f_list[id][2],
                "CH_2_f": stages_f_list[id][1],
                "CH_1_f": stages_f_list[id][0],

                "AM": stages_m_list[id][4],
                "PA_2_m": stages_m_list[id][3],
                "PA_1_m": stages_m_list[id][2],
                "CH_2_m": stages_m_list[id][1],
                "CH_1_m": stages_m_list[id][0],

                "CF": clean_fish_save_list[id]
        }

        sim.__dict__["Farm_%s" % id] = pd.DataFrame(data=d)
    return sim


