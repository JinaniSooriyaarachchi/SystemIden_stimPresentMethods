#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 20:44:19 2022

@author: jinani
"""

import scipy.io
import numpy as np

def generateDatasets(nMovies, nTrain, nValid,nTest, imgSiz, nFrames, rfModel,noise_level,iterations):
        
    nPixels  = np.power(imgSiz,2)    
    stimTrain=None
    respTrain=None
    stimTest=None
    respTest=None
    stimValid=None
    respValid=None


    for x in range(1,nMovies+1):

        filename="McGill_clips_0"+str(x)+".mat"
        print(filename)
        mat = scipy.io.loadmat(filename)
        stimulus=mat['thisNewMovie']
        mSiz=len(stimulus)
        if imgSiz>mSiz:
            print('Error:requested imgSiz is too large for this movie-set')
        elif imgSiz<mSiz:
            begX = round((mSiz-imgSiz)/2)
            endX = begX+imgSiz
            xMovie = stimulus[begX:endX,begX:endX,:]
        else:
            xMovie = stimulus
            
        stimMovie = xMovie.reshape(nPixels,nFrames)
        stimMovie = stimMovie.astype(np.float)
        stimMovie = stimMovie/128
        Response=np.zeros((iterations,1,375))
        for i in range(iterations):
            response1 = np.matmul(rfModel,stimMovie)+ noise_level*np.random.normal(size=(nFrames))
            ### Half wave rectifier remove all negative values
            response1=np.clip(response1, 0, None) 

            Response[i,:,:]=response1
            
        if iterations>1:    
            response=np.mean(Response,axis=0)
        else:
            response=np.squeeze(Response,axis=0)

        #### Create the datasets for training, validation and testing

        if x<= nTrain:
            if stimTrain is None:
                stimTrain=stimMovie                
            else:
                stimTrain = np.concatenate(([stimTrain , stimMovie ]), axis=1)
               
            if respTrain is None:
                respTrain=response
            else:
                respTrain = np.concatenate(([respTrain , response ]), axis=1)
       
        elif x<=nTrain+nValid:
            if stimValid is None:
                stimValid=stimMovie
               
            else:
                stimValid = np.concatenate(([stimValid , stimMovie ]), axis=1)
               
            if respValid is None:
                respValid=response
            else:
                respValid = np.concatenate(([respValid , response ]), axis=1)
        else:
            if stimTest is None:
                stimTest=stimMovie
               
            else:
                stimTest = np.concatenate(([stimTest , stimMovie ]), axis=1)
              
            if respTest is None:
                respTest=response
            else:
                respTest = np.concatenate(([respTest , response ]), axis=1)
        
    stimTrain=np.transpose(stimTrain)
    respTrain=np.transpose(respTrain)
    stimTest=np.transpose(stimTest)
    respTest=np.transpose(respTest)
    stimValid=np.transpose(stimValid)
    respValid=np.transpose(respValid)


    return stimTrain, stimTest, stimValid, respTrain, respTest, respValid,Response



