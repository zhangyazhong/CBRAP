# -*- coding: utf-8 -*-

import numpy as np

class load:
     def __init__(self,fbpath,ftpath):
         self.arm2features={}
         self.user2arms={}
         self.nbarms=0
         self.nbfeatures=0
         self.fbpath=fbpath
         self.ftpath=ftpath
         #print self.fbpath
         with open( self.ftpath, 'r') as f:
             for line in f:
                 lstr = line.strip().split()
                 self.nbfeatures=len(lstr)
                 self.arm2features[self.nbarms]=self.arm2features.get(self.nbarms,np.zeros(self.nbfeatures))    
                 self.arm2features[self.nbarms][0:len(lstr)] = [float(k) for k in lstr]
                 self.nbarms+=1
        
                 
         with open( self.fbpath, 'r') as f:
             for line in f:
                 lstr = line.strip().split()
                 u, i, r = [float(k) for k in lstr]
                 self.user2arms[u] = self.user2arms.get(u,dict())
                 self.user2arms[u][i] = r

         self.nbusers=len(self.user2arms.keys())