# -*- coding: utf-8 -*-
'''Generic index policy.'''

from random import choice

from policy.Policy import *

class IndexPolicy(Policy):
    """Class that implements a generic index policy."""

#  def __init__(self):

#  def computeIndex(self, arm):

    def choice(self):
        """In an index policy, choose at random an arm with maximal index."""
        index = dict()
        for arm in range(self.nbArms):
            index[arm] = self.computeIndex(arm, None)
        maxIndex = max (index.values())
        bestArms = [arm for arm in index.keys() if index[arm] == maxIndex]
        return choice(bestArms)

    def choicex(self,arms):        
        """In an index policy, choose at random an arm with maximal index."""
        index = dict()
        for arm in range(self.nbArms):
            index[arm] = self.computeIndex(arm,arms[arm])
        #print(index)
        
        maxIndex = max (index.values())
        bestArms = [arm for arm in index.keys() if index[arm] == maxIndex]
        return bestArms