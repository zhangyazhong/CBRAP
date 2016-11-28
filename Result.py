# -*- coding: utf-8 -*-
'''Utility class for handling the results of a Multi-armed Bandits experiment.'''

import numpy as np

class Result:
    """The Result class for analyzing the output of bandit experiments."""
    def __init__(self, nbArms, horizon):
        self.nbArms = nbArms
        self.choices = np.zeros(horizon)
        self.rewards = np.zeros(horizon)
    
    def store(self, t, choice, reward):
        self.choices[t] = choice
        self.rewards[t] = reward
    
    def getNbPulls(self):
        if (self.nbArms==float('inf')):
            self.nbPulls=np.array([])
            pass
        else :
            nbPulls = np.zeros(self.nbArms)
            for choice in self.choices:
                nbPulls[choice] += 1
            return nbPulls
    
    def getRegret(self, bestExpectation):
        return np.cumsum(bestExpectation-self.rewards)
        
    def __repr__(self):
        return "<Result choices:%s \n reward:%s>" % (self.choices, self.rewards)
