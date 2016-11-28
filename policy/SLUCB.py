# -*- coding: utf-8 -*-

import numpy as np
import random as rand
from policy.IndexPolicy import IndexPolicy

class SLUCB(IndexPolicy):
    """The SLUCB.
      Reference1: [Carpentier, Munos - AISTATS, 2012].
      Reference2: [Dani, Hayes and Kakade - COLT, 2008].
    """
    def __init__(self, nbArms, nbFeatures):
        self.nbArms = nbArms
        self.nbFeatures = nbFeatures
        self.thetaCB2 = np.zeros(self.nbFeatures)
        self.ACB2 = np.matrix(np.eye(self.nbFeatures))
        self.betaCB2 = 0.0
        self.sigma = 1
        self.theta = 1
        self.delta = 0.01
        self.T = 30
        self.Theta = {}
        for d in range(self.nbArms):
            self.Theta.update({d: np.matrix(np.zeros(self.nbFeatures))})
        
    def startGame(self, rounds, reductionDim):
        self.b = (self.theta + self.sigma)*np.sqrt(2*np.log(2*self.nbFeatures/self.delta))
        self.t = 0
        self.totalRounds = rounds
        
    def getReward(self, armid, arm, reward):
        self.t += 1
        if self.t <= self.T:
            if self.t == 1:
                if reward == 0:
                    reward = 0.1 # avoid singularity
                self.MatrixSEP = np.matrix(arm)
                self.VectorSEP = np.matrix(reward)
                
            else:
                self.MatrixSEP = np.vstack([self.MatrixSEP, np.matrix(arm)])
                self.VectorSEP = np.vstack([self.VectorSEP, np.matrix(reward)])
                
            self.Theta[armid] = np.matrix(self.nbFeatures*self.MatrixSEP.transpose()*self.VectorSEP/self.t).transpose()
            if (np.max(self.Theta[armid]) - np.min(self.Theta[armid])) != 0:
                self.Theta[armid] = (self.Theta[armid] - np.min(self.Theta[armid])) / (np.max(self.Theta[armid]) - np.min(self.Theta[armid]))
        else:
            if self.t == self.T + 1:
                self.Matrix = np.matrix(arm)
                self.Vector = np.matrix(reward)
            else:
                self.Matrix = np.vstack([self.Matrix, np.matrix(arm)])
                self.Vector = np.vstack([self.Vector, np.matrix(reward)])
            self.ACB2 = self.ACB2 + np.matrix(arm).transpose()*np.matrix(arm)
            self.thetaCB2 = np.linalg.inv(self.ACB2)*self.Matrix.transpose()*self.Vector
        
    def computeIndex(self, armid, arm):
        armThetaNorm = {}
        for d in range(self.nbArms):
            armThetaNorm.update({d: np.linalg.norm(self.Theta[d])})
            
        if self.t == 0:
            x = self.shiftZeroOne(self.generateRandomExplore()) # randomly pull an arm
            distance = np.linalg.norm(x-np.matrix(arm)) # select the arm with the smallest distance
            return 1.0/distance
        maxIndex = max(armThetaNorm.values()) # obtain the maximum theta norm
        
        if self.t <= self.T:
            while ((((maxIndex - 2*self.b/np.sqrt(self.t)) < 0) or (self.t < np.sqrt(self.totalRounds)/(maxIndex - self.b/np.sqrt(self.t))))):
                x = self.shiftZeroOne(self.generateRandomExplore()) # randomly pull an arm                             
                distance = np.linalg.norm(np.matrix(x)-np.matrix(arm)) # select the arm with the smallest distance
                return 1.0/distance
            return 0.0  
        else:                
            armSetA = {}
            for d in range(self.nbArms):
                if (np.linalg.norm(self.Theta[d]) >= 2*self.b/np.sqrt(self.T)):
                    armSetA.update({d:self.Theta[d]})
                    
            if bool(armSetA): # avoid empty set
                d = rand.sample(self.Theta.keys(),1)
                armSetA.update({d[0]:self.Theta[d[0]]})
                
            result  = self.computeIndexCB2(armid, arm, armSetA)
            return result
    
    def computeIndexCB2(self, armid, arm, armSetA):
        
        beta1 = 128*self.nbFeatures*np.log(self.t)*(np.log(np.square(self.t)/self.delta))
        beta2 = np.square(np.log(np.square(self.t)/self.delta)*8/3)
        beta = max([beta1,beta2])
        
        #beta = 128*self.nbFeatures*np.square((np.log(np.square(self.totalRounds)/self.delta)))
        Ball = {}
        for d in range(self.nbArms):
            if d in armSetA:
                if (np.linalg.norm(((np.matrix(armSetA[d]) - np.matrix(self.thetaCB2))*self.ACB2*((np.matrix(armSetA[d]) - np.matrix(self.thetaCB2)).transpose()))) <= np.sqrt(beta)):
                    Ball.update({d: armSetA[d]})
        if bool(Ball):
           d = rand.sample(armSetA.keys(),1) # avoid empty set
           Ball.update({d[0]: armSetA[d[0]]})
        
        v = np.matrix(np.zeros(self.nbFeatures))
        maxDot = 0.0
        for d in Ball:
            v = np.matrix(Ball[d])
            dotValue = np.linalg.norm(v*((np.matrix(arm)).transpose()))
            if maxDot < dotValue:
                maxDot = dotValue
                v = np.matrix(Ball[d]).copy()
        result = np.matrix(arm)*v.transpose()
        return result

    def generateRandomExplore(self):
        randomValue = np.random.rand(self.nbFeatures)
        exploreVector = np.zeros(self.nbFeatures)
        for pos in range(self.nbFeatures):
            value = randomValue[pos]
            if value < 0.5:
                exploreVector[pos] = -1.0/np.sqrt(self.nbFeatures)
            else:
                exploreVector[pos] = 1.0/np.sqrt(self.nbFeatures)
        return exploreVector       
        
    def shiftZeroOne(self, vector):
        maxValue = np.max(vector)
        minValue = np.min(vector)
        if (maxValue - minValue) != 0.0:
            vector = (vector - minValue)/(maxValue - minValue)
        else:
            vector = vector*0.0
        return vector
        
        