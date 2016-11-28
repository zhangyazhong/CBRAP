# -*- coding: utf-8 -*-

import numpy as np
import random as rand
from policy.IndexPolicy import IndexPolicy

class CBRAPSG(IndexPolicy):

    def __init__(self, nbArms, nbFeatures):
        self.nbArms = nbArms
        self.nbFeatures = nbFeatures
        self.alpha=0.1
        

    def startGame(self, rounds, reductionDim):
        self.newFeatures = reductionDim
        self.X=np.matrix(np.eye(self.newFeatures))
        self.b=np.matrix(np.zeros(self.newFeatures))
        self.z=np.matrix(np.zeros(self.newFeatures))
        self.R=np.matrix(np.zeros((self.newFeatures, self.nbFeatures)))
        self.theta=np.linalg.inv(self.X)*np.transpose(np.matrix(self.b))        
        #self.theta = np.linalg.inv((np.matrix(np.eye(self.newFeatures))/self.newFeatures + self.X.transpose()*self.X))*self.X.transpose()*np.transpose(np.matrix(self.b))# ridge regression
        
		## standard random matrix with N(0, 1)
        for i in range(self.newFeatures):
            for j in range(self.nbFeatures):
                randomValue = rand.gauss(0, 1)
                self.R.itemset(i, j, randomValue)   
        
    def getReward(self, armid, arm, reward):   
        self.z = self.R*np.matrix(arm).transpose()
        self.X+=self.z*self.z.transpose()
        self.b+=np.dot(self.z.transpose(), reward)
        self.theta=np.linalg.inv(self.X)*np.transpose(np.matrix(self.b))
       #self.theta = np.linalg.inv((np.matrix(np.eye(self.newFeatures))/self.newFeatures + self.X.transpose()*self.X))*self.X.transpose()*np.transpose(np.matrix(self.b)) # ridge regression
    
    def computeIndex(self, armid, arm):
        """
        arm is the contextual information
        """
        self.z = self.R*np.matrix(arm).transpose()
        result = (self.theta.transpose()*self.z).item(0,0)+self.alpha*np.sqrt((self.z.transpose()*np.linalg.inv(self.X)*self.z).item(0,0)) 
        return result