import numpy as np
from dataclasses import dataclass
from typing import List
from collections.abc import Iterable

@dataclass
class Food_treat:
    time:float 
    eff:np.array

@dataclass
class Treatment:
    '''define a single treatment'''
    _type: str
    time: float
    eff: np.array
    isfood: bool = False
    duration: float = 1

ANY = Treatment(
    _type='any',
    time=0,
    eff=np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    isfood=False,
    duration=1
)

TREAT_DICT = {
    'Diflubenzuron' : Treatment(
        _type='Diflubenzuron',
        time=0,
        eff=np.array([0.94, 0.94, 0.94, 0.94, 0.97, 0.97]),
        isfood=False,
        duration=1),
    'Slice' : Treatment(
        _type='Slice',
        time=0,
        eff=np.array([0.94, 0.94, 0.94, 0.94, 0.97, 0.97]),
        isfood=True,
        duration=40
    ),
    'Pyretroid' : Treatment(
        _type='Pyretroid',
        time=0,
        eff=np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        isfood=False,
        duration=1
    ),
    'H2O2' : Treatment(
        _type='H2O2',
        time=0,
        eff=np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        isfood=False,
        duration=1
    ),
    'any' : ANY
}

class Treatments_control:
    '''
    A class to manige the treatments in a Farm
    '''

    def __init__(self, treatments, NumTreat=0, treatment_type=None, treat_eff:np.array=None, treatment_period=None, is_food=None):

        self.NumTreat = NumTreat
        N = len(treatments)
        self.N = N
        self.num_treat_tjek = np.alen(treatments)

        treatments = list(treatments)

        #  TODO make None values in the list's be set to a default value
        if treatment_type is None:
            treatment_type = ['any' for _ in range(N)]

        treatment_type = list(treatment_type)

        if treat_eff is None:
            treat_eff = []
            for _type in treatment_type:
                temp = TREAT_DICT.get(_type, ANY)
                treat_eff.append(temp.eff)
            treat_eff = np.array(treat_eff).T

        if treat_eff.ndim == 2:
            shape = treat_eff.shape
            if shape == (6, N):
                treat_eff = treat_eff.T
            elif shape == (N, 6):
                pass
            else:
                raise ValueError
        else:
            raise ValueError

        if isinstance(treatment_period, int):
            treatment_period = [treatment_period for _ in range(N)]

        elif treatment_period is None:
            treatment_period = []
            for _type in treatment_type:
                temp = TREAT_DICT.get(_type, ANY)
                treatment_period.append(temp.duration)

        if isinstance(is_food, bool):
            is_food = [is_food for _ in range(N)]

        #  TODO TREAT_DICT shall take priorety
        elif is_food is None:
            is_food = []
            for _type in treatment_type:
                temp = TREAT_DICT.get(_type, ANY)
                is_food.append(temp.isfood)

        treat = []
        for _type, time, eff, food, periode in zip(
            treatment_type, treatments, treat_eff, is_food, treatment_period
        ):
            treat.append(
                Treatment(
                    _type = _type,
                    time = time,
                    eff = np.array(eff),
                    isfood = food,
                    duration = periode
                )
            )

        treat.sort( key = lambda x: x.time)
        self.treat = treat
        self.food_dict = {}

        self.done = False
        try:
            self.current_treat = self.treat[self.NumTreat]
        except IndexError:
            self.done = True

        while (not self.done) and (self.current_treat.time < 0 or np.isnan(self.current_treat.time)):
            if self.current_treat.isfood:
                if self.current_treat.time + self.current_treat.duration>0:
                    self.food_dict[self.current_treat._type] = \
                            self.current_treat.time + self.current_treat.duration
            self.NumTreat += 1
            try:
                self.current_treat = self.treat[self.NumTreat]
            except IndexError:
                self.done = True
                break


    def apply_Treat(self, time, dt):

        no_treat = True
        eff = np.ones(6, dtype=np.float64)

        while not self.done:

            if time >= self.current_treat.time:

                if self.current_treat.isfood:
                    self.food_dict[
                        self.current_treat._type
                    ] = Food_treat(
                        time = self.current_treat.duration,
                        eff = self.current_treat.eff**dt
                    )
                else:
                    eff *= self.current_treat.eff
                    no_treat = False

                self.NumTreat += 1
                try:
                    self.current_treat = self.treat[self.NumTreat]
                except IndexError:
                    self.done = True

            elif np.isnan(self.current_treat.time):
                self.NumTreat += 1
                try:
                    self.current_treat = self.treat[self.NumTreat]
                except IndexError:
                    self.done = True
            else:
                break

        time_to_next = np.inf if self.done else self.current_treat.time - time
        for key in list(self.food_dict):
            eff *= self.food_dict[key].eff
            self.food_dict[key].time -= dt

            if self.food_dict[key].time <= 0:
                self.food_dict.pop(key)
            else:
                time_to_next = 0
            no_treat = False

        if no_treat:
            return False, None, time_to_next
        return True, eff, time_to_next
