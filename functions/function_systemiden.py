#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 18:17:54 2022

@author: jinani
"""

from tensorflow import keras
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2


def sysiden(stimTrain,respTrain,stimValid,respValid,stimTest,respTest,imgSiz,l2_value):  
         
    visual_stim = keras.layers.Input(shape=[1024], name="visual_stim")
    stim_driven_unit = keras.layers.Dense(1,kernel_regularizer=l2(l2_value),activation="relu", name="stim_driven_unit")(visual_stim)
    model = keras.Model(inputs=[visual_stim], outputs=[stim_driven_unit])
    model.summary()
    
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1)
    # fit model
    model.fit(stimTrain, respTrain, epochs=1000, batch_size=125, verbose=0, validation_data=(stimValid, respValid),callbacks=[es,mc])
         
    # predict on the test dataset
    predicted_test_response = model.predict(stimTest)
    predicted_valid_response = model.predict(stimValid)
    
    # calculate test VAF
    predicted_test_response = predicted_test_response.reshape(-1)
    respTest=respTest.reshape(-1)
    R=np.corrcoef(predicted_test_response,respTest)
    diag=R[0,1]
    vaf_test=diag*diag*100
    
    # calculate valid VAF
    predicted_valid_response = predicted_valid_response.reshape(-1)
    respValid=respValid.reshape(-1)
    R=np.corrcoef(predicted_valid_response,respValid)
    diag=R[0,1]
    vaf_valid=diag*diag*100
   
    # get the weights and biases in each layer
    weights = model.get_weights()
    
    # plot the estimated RF model
    rfEstimate=(weights[0])
    RF = rfEstimate.reshape(imgSiz,imgSiz)
    
    
    
    return vaf_test,vaf_valid, RF, predicted_test_response
    
    
    
    
    
    