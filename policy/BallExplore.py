# -*- coding: utf-8 -*-

import numpy as np
import random as rand
from policy.IndexPolicy import IndexPolicy

class BallExplore(IndexPolicy):
    """The BallExplore.
      Reference: [Deshpande and Montanari - AAC, 2012].
    """
    def __init__(self, nbArms, nbFeatures):
        self.nbArms = nbArms
        self.nbFeatures = nbFeatures
        self.theta = np.matrix(np.zeros(self.nbFeatures))   
    
    def startGame(self, rounds, reductionDim):
        self.P = np.matrix(np.eye(self.nbFeatures))
        self.Delta = 1.0
        self.t = 1
        
    def getReward(self, armid, arm, reward):
        if self.t == 1:
            if reward == 0.0:
                reward = 0.1 # avoid singularity
            self.Matrix = np.matrix(arm)
            self.Vector = np.matrix(reward)
        else:
            self.Matrix = np.vstack([self.Matrix, np.matrix(arm)])
            self.Vector = np.vstack([self.Vector, np.matrix(reward)])
        self.Matrix = (self.Matrix - np.mean(self.Matrix)) / np.std(self.Matrix) # standardize the input matrix
        self.t = self.t + 1
        self.sigma = self.Delta/self.nbFeatures
        self.theta = np.linalg.inv((np.matrix(np.eye(self.nbFeatures))/self.nbFeatures + self.Matrix.transpose()*self.Matrix))*self.Matrix.transpose()*self.Vector/self.sigma # ridge regression
        self.P = np.matrix(np.eye(self.nbFeatures)) - self.theta.transpose()*self.theta/np.square(self.theta)
            
    def computeIndex(self, armid, arm):
        if np.linalg.norm(self.theta) == 0.0:
            self.xx = np.matrix(np.zeros(self.nbFeatures))
        else:
            self.xx = (self.theta)/np.linalg.norm(self.theta)
        
        self.u = np.random.rand(self.nbFeatures)
        self.uu = np.matrix(self.u/np.linalg.norm(self.u))
        
        self.beta = np.sqrt(2.0/3.0)*np.sqrt(np.sqrt(np.min([self.nbFeatures*self.Delta/self.t,1.0])))

        self.x = np.sqrt(1-np.square(self.beta))*self.xx + (self.beta*self.P*self.uu.transpose()).transpose()
        
        if np.isnan(self.x.all()):
            distance = rand.random() # detect some nan case, just choose random value
        else:
            distance = np.linalg.norm(self.x-np.matrix(arm)) # select the arm with the smallest distance
        
        return 1.0/distance
