# -*- coding: utf-8 -*-
'''A utility class for evaluating the performance of a policy in multi-armed bandit problems.'''

import numpy as np

class Evaluation:
  
    def __init__(self, env, pol, nbRepetitions, horizon, reductionDim, tsav=[]):
        if len(tsav)>0:
            self.tsav = tsav
        else:
            self.tsav = np.arange(horizon)
        self.env = env
        self.pol = pol
        self.nbRepetitions = nbRepetitions
        self.horizon = horizon
        self.nbArms = env.nbArms
        self.nbPulls = np.zeros((self.nbRepetitions, self.nbArms))
        self.cumReward = np.zeros((self.nbRepetitions, self.horizon))
        self.reductionDim = reductionDim

        # progress = ProgressBar()
        for k in range(nbRepetitions): # progress(range(nbRepetitions)):
            result = env.play(pol, horizon, self.reductionDim)
            
            #print len(np.cumsum(result.rewards))            
            self.nbPulls[k,:] = result.getNbPulls()
            self.cumReward[k] = np.cumsum(result.rewards,dtype=float)
        #print(self.cumReward)
        # progress.finish()
     
    def meanReward(self):
        return sum(self.cumReward[:,-1])/len(self.cumReward[:,-1])
    
    def cumuwards(self):
        return np.cumsum(self.cumReward,axis=0)[self.nbRepetitions-1,:]/self.nbRepetitions  

    def meanNbDraws(self):
        return np.mean(self.nbPulls ,0) 

    def meanRegret(self):
        return (1+self.tsav)*1 - np.mean(self.cumReward, 0)

    def allCumReward(self):
        return self.cumReward
        