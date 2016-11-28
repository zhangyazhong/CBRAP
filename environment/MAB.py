# -*- coding: utf-8 -*-
'''
Environement for a Multi-armed bandit problem with arms given in the 'arms' list 
'''

from Result import *
from environment.Environment import Environment
import numpy as np

class MAB(Environment):
    """Multi-armed bandit problem with arms given in the 'arms' list"""
    
    def __init__(self, arms,truth):
        self.arms = arms
        self.nbArms = len(arms)
        self.truth=truth
        # supposed to have a property nbArms
    
    def play(self, policy, horizon, reductionDim):
        policy.startGame(horizon, reductionDim)
        result = Result(self.nbArms, horizon)
        for t in range(horizon):       
            if self.truth==None:
                choice = policy.choice()
                
                reward=self.arms[choice].draw()
                policy.getReward(choice, None, reward)
                result.store(t, choice, reward)
            else:
                choice = policy.choicex(self.arms)
                f_c = np.random.choice(choice)
                if f_c in self.truth:
                    reward = self.truth[f_c] # for ranking else reward = 1
                else:
                    reward = 0 
                policy.getReward(f_c, self.arms[f_c], reward)
                result.store(t, f_c, reward)
        return result
   
   
   
