# -*- coding: utf-8 -*-
"""
@author: Harnick Khera (Github.com/Hephyrius)
@author: Vitaly Feskov (Github.com/vftens)

Use this class to train a basic machine learning model. Or modify it to incoportate your own machine learning models or Pipelines by
using other libraries such as XGBoost, Keras or strategies like Spiking Neural Networks!

"""

from cython_cppwraper.mainrrl import NeuralRRL

import numpy as np
from numpy import *
import pandas as pd

from binance.client import Client
from binance.enums import *


import CoreFunctions as cf

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from joblib import dump, load

#You don't need to enter your key/secret in order to get data from the exchange, its only needed for trades in the TradingBot.py class.
api_key = '0'
api_secret = '0'
client = Client(api_key, api_secret)

print("Loading candles from Binance...")
candles = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_5MINUTE, "01 Jan, 2017", "23 Jul, 2021")

print("PREPARING Data...")
##Convert the raw data from the exchange into a friendlier form with some basic feature creation
x = cf.FeatureCreation(candles)
#
##Create our targets
y = cf.CreateTargets(candles,1)
#
##remove the top elements of the features and targets - this is for certain features that arent compatible with the top most
##for example SMA27 would have 27 entries that would be incompatible/incomplete and would need to be discarded
y = y[94:]
x = x[94:len(candles)-1]
#
##produce sets, avoiding overlaps!
##data is seporated temporily rather than randomly
##this prevents the model learning stuff it wouldnt know - aka leakage - which can give us false positive models
trny = y[:9999]
trnx = x[:9999]
#
##Validation set is not used in this starter model, but should be used if using other libraries that support early stopping.
valy = y[10000:12999]
valx = x[10000:12999]
#
tsty = y[13000:]
tstx = x[13000:]

print("Loading GradientBoostingClassifier...")
#
model = GradientBoostingClassifier() 
model.fit(trnx,trny)
#
preds = model.predict(tstx)
#
##Some basic tests so we know how well our model performs on unseen - "modern" data.
##Helps with fine tuning features and model parameters
accuracy = accuracy_score(tsty, preds)
mse = mean_squared_error(tsty, preds)
#
print("Accuracy = " + str(accuracy))
print("MSE = " + str(mse))
#
#falsePos = 0
#falseNeg = 0
#truePos = 0
#trueNeg = 0
#total = len(preds)
#
#for i in range(len(preds)):
#    
#    if preds[i] == tsty[i] and tsty[i] == 1:
#        truePos +=1
#        
#    elif preds[i] == tsty[i] and tsty[i] == 0:
#        trueNeg +=1
#        
#    elif preds[i] != tsty[i] and tsty[i] == 1:
#        falsePos +=1
#        
#    elif preds[i] != tsty[i] and tsty[i] == 0:
#        falseNeg +=1
#        
#print("False Pos = " + str(falsePos/total))
#print("False Neg = " + str(falseNeg/total))
#print("True Pos = " + str(truePos/total))
#print("True Neg = " + str(trueNeg/total))
#
##how important of the features - helps with feature selection and creation!
#results = pd.DataFrame()
#results['names'] = trnx.columns
#results['importance'] = model.feature_importances_
#print(results.head)
#
#
##save our model to the system for use in the bot
#dump(model, open("Models/model.mdl", 'wb'))








