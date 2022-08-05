#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 20:52:07 2022

@author: jinani


This code studies the trade-off between the number of movies/stimuli used for 
system identification and the number of repetitions the movie was presented. 
System identification is done with a simple machine learning model.
This is one of the main concepts that should be planned during the planning 
stage of electrophysiology experiments.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'  
from IPython import get_ipython
get_ipython().magic('reset -sf') #to Clear Variables Before Script Runs

from functionGenerateModelRF import modelRF
from functionGenerateDatasets_withRepetitions import generateDatasets
from function_systemiden import sysiden
import numpy as np
import matplotlib.pyplot as plt

## Input arguments for functions modelRF and tempFilter
carrLambda = 8
carrOri    = 0
imgSiz=32
nValid=1 
nTest=1
nFrames=375 
nPixels  = np.power(imgSiz,2)
noise_level=2 # noise=0 or noise =1
l2_value=0.01 # this is a pre selected L2 value.

rf_Plot, rfModel=modelRF(carrLambda,carrOri,imgSiz)

Num_Movies=[1,2,3,4]
ITR=[1,2,3,4,5]
vaf=[]
rf=[]
for nTrain in Num_Movies:
    for iterations in ITR:
        nMovies=nTrain+nValid+nTest
        stimTrain, stimTest, stimValid, respTrain, respTest, respValid,Response=generateDatasets(nMovies, nTrain, nValid,nTest, imgSiz, nFrames, rfModel,noise_level,iterations)
        
        vaf_test, vaf_valid, RF, predicted_test_response = sysiden(stimTrain,respTrain,stimValid,respValid,stimTest,respTest,imgSiz,l2_value)
        
        vaf.append(vaf_test)
        rf.append(RF)

error=[]
for i in range(20):
    RF=np.reshape(rf[i], (1,1024))  
    error.append(np.mean((RF-rfModel)**2))
   
plt.figure()
for mov in range (4):
    labl=str(mov+1) + '  Movies'
    plt.plot(ITR,error[(mov)*5:(mov+1)*5],label=labl)  
    plt.legend('movie')
plt.grid()  
plt.legend(loc="upper right")
plt.title('Mean squared error between the actual and estimated RF')
plt.ylabel('Mean squared error')
plt.xlabel('Repetitions')
plt.xticks(ITR) 

plt.figure()
for mov in range (4):
    labl=str(mov+1) + '  Movies'
    plt.plot(ITR,vaf[(mov)*5:(mov+1)*5],label=labl)  
    plt.legend('movie')
plt.grid()  
plt.legend(loc="lower right")
plt.title('Comparison of VAFs')
plt.ylabel('VAF')
plt.xlabel('Repetitions')
plt.xticks(ITR) 
 
fig=plt.figure()          
for i in range(20):
    plt.subplot(4,5,i+1)
    plt.imshow(rf[i])
    plt.xticks([])
    plt.yticks([])
fig.text(0.5, 0.04, 'Repetitions (1 to 5)', ha='center')
fig.text(0.04, 0.5, 'Number of movies (top to bottom = 1 to 5)', va='center', rotation='vertical')
plt.suptitle('Estimated Receptive fields')



    