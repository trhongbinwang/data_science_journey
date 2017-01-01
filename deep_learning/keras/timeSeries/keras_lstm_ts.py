# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 14:41:34 2016

keras time series prediction with window past

tuning performance still an unsolved issue

@author: hongbin
"""

import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Model
from keras.layers import Input, Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# fix random seed for reproducibility
numpy.random.seed(7)

# hyperparameters
look_back = 10 # window size of past


def create_dataset(dataset, look_back=1):
# convert an array of values into a dataset matrix
    
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

def Normalize_data(dataset):
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    return [dataset, scaler]
    

def pre_processing(dataset):
    ''' 
    normalize, split and reshape the dataset
    '''
    [dataset, scaler] = Normalize_data(dataset)
    [train, test] = split_data(dataset)
    [trainX, trainY, testX, testY] = reshape_data(train, test)
    print(trainX.shape, testX.shape)
    return [trainX, trainY, testX, testY, scaler]

def split_data(dataset):
    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    return [train, test]

def reshape_data(train, test):
    # reshape into X=t and Y=t+1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    return [trainX, trainY, testX, testY]
    

# load the dataset
def load_data():
    '''
    load data
    '''
    dataframe = pandas.read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    print(dataset[0:5,:])
    return dataset
    
def model():
    ''' 
    Input is in keras/engine/topology.py
    '''
    # Input is use to create a keras tensor. check how keras use with tf
    x = Input(shape=(1,look_back)) # (94, 1, 1) shape is the training set's shape
    l = LSTM(64)(x)
    d = Dense(1)(l)
    model = Model(input=x, output=d)
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model
    
def predict(model, trainX, testX):
    ''' '''
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    return [trainPredict, testPredict]
    
    
def invert_predict(trainPredict, trainY, testPredict, testY):
    
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    return [trainPredict, trainY, testPredict, testY]
    
def get_score(trainPredict, trainY, testPredict, testY):    
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))

def display(dataset, trainPredict, testPredict):
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
    # plot baseline and predictions
    plt.plot(dataset)
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()


if __name__ == '__main__':
    ''' '''
    # load data
    dataset = load_data()
    # pre-porcessing. scaler will use later
    [trainX, trainY, testX, testY, scaler] = pre_processing(dataset)
    # define the model
    lstm_ts = model()
    # train the model
    lstm_ts.fit(trainX, trainY, nb_epoch=20, batch_size=1, verbose=2)
    # predict
    [trainPredict, testPredict] = predict(lstm_ts, trainX, testX)
    # invert predictions 
    [trainPredict, trainY, testPredict, testY] = invert_predict(trainPredict, trainY, testPredict, testY)
    # calculate root mean squared error 
    get_score(trainPredict, trainY, testPredict, testY)
    # plot the result
    display(dataset, trainPredict, testPredict)








