#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 09:17:42 2021

@author: jinani
"""

"""
This function will create a gabor filter.
"""
import math
import numpy as np
import matplotlib.pyplot as plt

## create the gabor model RF


#carrLambda = 8
#carrOri    = -30
#imgSiz=32

def modelRF(carrLambda,carrOri,imgSiz):

    pi=math.pi
    nPixels  = np.power(imgSiz,2)
    nPts = imgSiz      
    carrSF     = 1 / carrLambda
    envSigX     = carrLambda/3
    envSigY     = envSigX
    carrOriRad = (pi/180)*carrOri
    carrPh      = 0   
    carrPhRad  = (pi/180)*carrPh
    envOri      = 0  
    envOriRad  = (pi/180)*envOri
    
    f = np.zeros((nPts,nPts))
    for x in range(0, nPts):
       for y in range(0, nPts):
       	xx = (x-nPts/2)*math.cos(envOriRad) - (y-nPts/2)*math.sin(envOriRad)
       	yy = (x-nPts/2)*math.sin(envOriRad) + (y-nPts/2)*math.cos(envOriRad)
       	env = math.exp( -np.power(xx,2)/(2*envSigX*envSigX) - np.power(yy,2)/(2*envSigY*envSigY) )
       	carr = math.sin(2*pi*carrSF*(x*math.cos(carrOriRad)+y*math.sin(carrOriRad)) + carrPhRad)
       	f[y][x] = env * carr
    
    #print(f)
    #print(np.shape(f))
    plt.imshow(f)
    plt.colorbar()
    rfModel = f.reshape(1,nPixels)
    plt.title("Actual RF")
    #print(np.shape(rfModel))
    
    return f, rfModel