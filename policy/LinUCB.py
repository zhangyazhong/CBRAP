# -*- coding: utf-8 -*-

import numpy as np
from policy.IndexPolicy import IndexPolicy

class LinUCB(IndexPolicy):
    """The LinUCB.
      Reference1: [Li, Chu, Langford and Schapire - WWW, 2010].
      Reference2: [Chu, Li, Reyzin and Schapire - AISTATS, 2011].
    """

    def __init__(self, nbArms, nbFeatures):
        self.lambda1=0.5
        self.nbArms = nbArms
        self.X=np.matrix(np.eye(nbFeatures))*self.lambda1
        self.b=np.matrix(np.zeros(nbFeatures))
        self.alpha=0.1

    def startGame(self, rounds, reductionDim):
        self.theta=np.linalg.inv(self.X)*np.transpose(np.matrix(self.b))

    def getReward(self, armid, arm, reward):
        self.X+=np.matrix(arm).transpose()*np.matrix(arm)
        self.b+=np.dot(np.matrix(arm),reward)
        self.theta=np.linalg.inv(self.X)*np.transpose(np.matrix(self.b))

    
    def computeIndex(self, armid,arm):    
        result= (self.theta.transpose()*np.matrix(arm).transpose()).item(0,0)+self.alpha*np.sqrt((np.matrix(arm)*np.linalg.inv(self.X)*np.matrix(arm).transpose()).item(0,0)) 
        return result