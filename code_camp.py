# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 13:39:56 2018

@author: siit
"""

import numpy
import matplotlib.pyplot as plt
import pandas
import math
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# convert an array of values into a dataset matrix

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back-1):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 18])
    return numpy.array(dataX), numpy.array(dataY)

# fix random seed for reproducibility
numpy.random.seed(7)

writer = pd.ExcelWriter('DBN & RNN Data/SatRNN1.xlsx')
for sheetNum in range(1,49):
    dataframe = pandas.read_excel('DBN & RNN Data/Sat.xlsx',sheet_name='Sheet'+str(sheetNum))
    
    # load the dataset
    dataset = dataframe.values
    
    #************************************************
    
    train_size = int(len(dataset) * 0.48)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    
    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    
    #***********************************************
    
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    
    # split into train and test sets
    train_size = int(len(dataset) * 0.48)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    
    
    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    
    # reshape input to be  [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], look_back, 19))
    testX = numpy.reshape(testX, (testX.shape[0],look_back, 19))
    
    # create and fit the LSTM network
    
    model = Sequential()
    model.add(LSTM(4, input_shape=(look_back,19)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    history= model.fit(trainX, trainY, epochs=200, batch_size=32)
    
    
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    
    # Get something which has as many features as dataset
    trainPredict_extended = numpy.zeros((len(trainPredict),19))
    # Put the predictions there
    trainPredict_extended[:,18] = trainPredict[:,0]
    # Inverse transform it and select the 5rd column.
    trainPredict = scaler.inverse_transform(trainPredict_extended) [:,18]
    print('trainPredict',trainPredict)
    # Get something which has as many features as dataset
    testPredict_extended = numpy.zeros((len(testPredict),19))
    # Put the predictions there
    testPredict_extended[:,18] = testPredict[:,0]
    # Inverse transform it and select the 5rd column.
    testPredict = scaler.inverse_transform(testPredict_extended)[:,18]
    print('testPredict',testPredict)
    
    trainY_extended = numpy.zeros((len(trainY),19))
    trainY_extended[:,18]=trainY
    trainY=scaler.inverse_transform(trainY_extended)[:,18]
    
    
    testY_extended = numpy.zeros((len(testY),19))
    testY_extended[:,18]=testY
    testY=scaler.inverse_transform(testY_extended)[:,18]
    
    
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY, testPredict))
    print('Test Score: %.2f RMSE' % (testScore))
    
    #*********************************************
    #calculate mean absolute percent error
    trainMAPE = mean_absolute_error(trainY, trainPredict)
    print('testMAPE: %.2f MAPE' % trainMAPE)
    testMAPE = mean_absolute_error(testY, testPredict)
    print('testMAPE: %.2f MAPE' % testMAPE)
    
    #calculate mean square error
    trainmse = mean_squared_error(trainY, trainPredict)
    print('TrainMSE: %.2f MSE' % trainmse)
    testmse = mean_squared_error(testY, testPredict)
    print('TrainMSE: %.2f MSE' % testmse)
    
    trainKPI = [];
    trainKPI.extend([trainMAPE,trainScore,trainmse])
    testKPI = [];
    testKPI.extend([testMAPE,testScore,testmse])
    print('trainKPI',trainKPI)
    print('testKPI',testKPI)
    #***************************************************
    
    print('train_size',train_size)
    print('test_size',test_size)
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, 18] = trainPredict
    
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, 18] = testPredict
    
    ## convert your array into a dataframe
    df = pd.DataFrame (trainPredict)
    dfTest = pd.DataFrame (testPredict)    
    dfTest.to_excel(writer,'Sheet'+str(sheetNum))
       
    print("================ DONE for sheet: "+str(sheetNum))
writer.save()
