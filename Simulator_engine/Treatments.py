import numpy as np

class Treatments:
    '''
    A class to manige the treatments in a Farm
    '''

    def __init__(self, treatments, NumTreat=0, treatment_type=None, treat_eff=None, treatment_period=None):

        try:
            treatments = list(treatments)
        except:
            raise TypeError(f"treatments must be a list'like object, I got {treatments}")

        self.treatment = treatments
        self.SliceON = 0 # -1 merkir at Slice ella Foodtreatment aðrar treatments ikk eru ON. > 0 eru ON
        self.NumTreat = NumTreat
        self.treatment_type = treatment_type
        self.treat_eff = np.array(treat_eff)
        self.num_treat_tjek = np.alen(treatments)-1
        self.treatment_period = treatment_period

    #  TODO it shuod be popsible to apply 2 treatments the same day
    def apply_Treat(self, time, dt):
        no_treat = True
        eff = np.ones(6, dtype=np.float64)
        if self.NumTreat<=self.num_treat_tjek:
            if time >= self.treatment[self.NumTreat]: # and self.treatment[self.NumTreat]>0:
                # if the treatment is after start of production cycle
                if self.treatment[self.NumTreat] > 0:
                    if self.treatment_type[self.NumTreat] in ['FoodTreatment:', 'Emamectin', 'Slice']:
                        self.SliceON = self.treatment_period # length of treatment effect
                        # NEW
                        self.Sliceeff = self.treat_eff[:, self.NumTreat]**dt
                        no_treat = False
                    else:
                        eff *= self.treat_eff[:, NumTreat]

                self.NumTreat += 1

            elif np.isnan(self.treatment[self.NumTreat]):
                self.NumTreat += 1

        #  TODO there can only be one Slice at a time
        if self.SliceON > 0:
            eff *= self.Sliceeff
            self.SliceON -= dt
            no_treat = False

        if no_treat:
            return False
        return eff

    #  TODO make a auto delicing
