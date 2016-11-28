# -*- coding: utf-8 -*-

import datetime # to show the date and time
import timeit # to calculate the time consumption
import random as rand

from environment.MAB import MAB
from numpy import *
from matplotlib.pyplot import *

from policy.SLUCB import SLUCB
from policy.BallExplore import BallExplore
from policy.CBRAPRS import CBRAPRS
from policy.CBRAPSG import CBRAPSG
from policy.LinUCB import LinUCB

from Evaluation import *
from prior.load import load

# figure setting
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'black', 'yellow', 'red']
markers = ['+','o','>','<','^','v','*','x'] 
graphic = 'no'

# running setting
nbRep = 3# 10
horizon = 100 #1000
reductionVector = [10]
for i in range(len(reductionVector)):
    
    reductionDim = reductionVector[i]

    fbfile="simulation/fb.txt"
    ftfile="simulation/ft.txt"
    data=load(fbfile,ftfile)
    print('#users: '+str(data.nbusers))
    print('#arms: '+str(data.nbarms))
    print('#reductionDim: '+str(reductionDim))
    print('#features: '+str(data.nbfeatures))
    print('#feedbacks: '+str(sum(len(data.user2arms.get(i)) for i in data.user2arms.keys())))
    print('#rounds: '+str(horizon))
    print('#Repetitions: '+str(nbRep))
    trunc = 10
    print(np.array(data.user2arms.keys()))
    
    sampled_u = rand.sample(data.user2arms.keys(),1)
    #sampled_u = [5]
    env = MAB(data.arm2features, data.user2arms[sampled_u[0]])
    K = env.nbArms
    #policies = [SLUCB(K, data.nbfeatures), BallExplore(K, data.nbfeatures), CBRAPSG(K, data.nbfeatures), CBRAPRS(K, data.nbfeatures), LinUCB(K, data.nbfeatures)]
    policies = [SLUCB(K, data.nbfeatures), BallExplore(K, data.nbfeatures), CBRAPSG(K, data.nbfeatures), CBRAPRS(K, data.nbfeatures)]
    tsav = int_(linspace(10,horizon-1,20))
    
    if graphic == 'yes':
        figure(1)
        
    fileCumu = 'MAB_cumu_reward_out'+'#users'+str(data.nbusers)+'#arms'+str(data.nbarms)+'#features'+str(data.nbfeatures)+'#rounds'+str(horizon)+'#Repetitions'+str(nbRep)+'#reductionDim'+str(reductionDim)
    fileTime = 'Consuming_time'+'#users''#users'+str(data.nbusers)+'#arms'+str(data.nbarms)+'#features'+str(data.nbfeatures)+'#rounds'+str(horizon)+'#Repetitions'+str(nbRep)+'#reductionDim'+str(reductionDim)
    fileAllCumu = 'All_cum_reward'+'#users'+str(data.nbusers)+'#arms'+str(data.nbarms)+'#features'+str(data.nbfeatures)+'#rounds'+str(horizon)+'#Repetitions'+str(nbRep)+'#reductionDim'+str(reductionDim)
    
    outfile = open('results/'+fileCumu,'w')
    outfileTime = open('results/'+fileTime, 'w')
    outfileAllCumu = open('results/'+fileAllCumu, 'w')
    
    k=0
    for policy in policies:
        print(policy)
        print(datetime.datetime.now())
        timeBegin = timeit.default_timer()
        ev = Evaluation(env, policy, nbRep, horizon, reductionDim, tsav)
        timeEnd = timeit.default_timer()
        dataAllCumu = ev.allCumReward()
        print(datetime.datetime.now())
        print(ev.meanReward())
        
        stdvar = np.sqrt(np.var(dataAllCumu, axis=0))
        cumuwards = np.array(ev.cumuwards())
        outfile.write(str(policy)+'\n')
        outfile.write(' '.join([str(x) for x in cumuwards]))
        outfile.write('\n')
        outfile.write(' '.join([str(x) for x in stdvar]))
        outfile.write('\n')
        
        outfileTime.write(str(policy)+'\n')
        outfileTime.write(str(timeEnd - timeBegin))
        outfileTime.write('\n')
        
        outfileAllCumu.write(str(policy)+'\n')
        for i in range(nbRep):
            dataCumu = dataAllCumu[i, :]
            outfileAllCumu.write(' '.join([str(x) for x in dataCumu]))
            outfileAllCumu.write('\n')
        
        outfileAllCumu.write(' '.join([str(x) for x in stdvar]))
        outfileAllCumu.write('\n')
        # plot figure
        if graphic == 'yes':
            ax = gca()
            semilogx(np.array(range(horizon)), cumuwards, color = colors[k], marker = markers[k])
            xlabel('Rounds')
            ylabel('Cumulative Rewards')
            ax.set_ylim(ymax=horizon*nbRep, ymin=-2)
            ax.set_xlim(xmax=horizon)
        k = k+1
    
    outfile.close()
    outfileTime.close()
    outfileAllCumu.close()
    
    if graphic == 'yes':
        legend([policy.__class__.__name__ for policy in policies], loc=0)
        title('Cumulative rewards')
        savefig('results/'+fileCumu+'.png',dpi = 500)
        show()
