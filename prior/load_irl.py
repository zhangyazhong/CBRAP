# -*- coding: utf-8 -*-

class load_irl:
    def __init__(self,ftpath):
        self.user2arms={}
        self.nbarms=0
        self.nbfeatures=0
        self.ftpath=ftpath
        #print self.fbpath
        self.nrecord=-1
        with open( self.ftpath, 'r') as f:
            for line in f:
                if self.nrecord==-1:
                    self.nrecord+=1
                    continue   
                lstr = line.strip().split()
                user=lstr[0]
                time=lstr[1]
                #print user
                self.user2arms[user]=self.user2arms.get(user,{})
                self.user2arms[user][time]=self.user2arms[user].get(time,[])
                self.user2arms[user][time]=[float(x) for x in lstr[2:]]
                #print self.user2arms[user][time]
                self.nbfeatures=len(lstr)-2
                self.nrecord+=1
        self.nbusers=len(self.user2arms.keys())
        #print self.nbusers
        self.avg_arm=0
        self.cumu_purchase=0
        for user in self.user2arms.keys():
            self.avg_arm+=len(self.user2arms.get(user))
            for arm in self.user2arms[user].keys():
                if int(self.user2arms[user][arm][-1])==1:
                    self.cumu_purchase+=1
        self.avg_purchase=float(self.cumu_purchase)/self.nbusers
        self.avg_arm/=self.nbusers
        
