''' Importing necessary libraries '''

# Data Manipulation 
import pandas as pd 
import numpy as np

# Visualisation libraries 
import matplotlib.pyplot as plt
import seaborn as sns

# Deep Learning libraries
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Extracting the data and building the train and test set as well as scaling them 

''' FUNCTION PARAMETERS - (dataset, 
feature_col: the column of the dataset being used for training the model, 
ratio: ratio of the train test split) ''' 

def train_test_data(dataset, feature_col, ratio):
    feature_data = dataset[[feature_col]].dropna().iloc[:, 0:1].values

    # Creating the training and testing data
    train_len = int(np.ceil(len(feature_data) * ratio))
    train_set = feature_data[:train_len]
    test_set = feature_data[train_len:]

    # Displaying info of data
    print("The size of the training set = {}".format(len(train_set)))
    print("The size of the test set = {}".format(len(test_set)))

    # Scaling the datasets
    scaler = MinMaxScaler(feature_range = (0, 1))
    scaled_train_set = scaler.fit_transform(train_set)

    # Returning the train and test sets 
    return scaled_train_set, test_set


# Timestep array function

''' FUNCTION PARAMETERS - (num_timesteps: number of previous days used for the feedback loop)'''

def timestep_array(num_timesteps):

    # Creating a data structure with 60 timesteps and 1 output
    # List containing data for every 60 days  
    X_train = []
    y_train = []

    for i in range(num_timesteps, len(train_set)):
        X_train.append(scaled_data[i - num_timesteps : i, 0])
        y_train.append(scaled_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    return X_train, y_train


# Simple function to reshape the input 

''' FUNCTION PARAMETERS - (data,  
num_timesteps: number of previous days used for the feedback loop,
batch_size: size of each batch,
input_dim: number of predictors'''

def reshape_input(data, num_timesteps, batch_size, input_dim):
    timesteps = num_timesteps
    no_of_predictors = input_dim
    X_train = np.reshape(X_train, (batch_size, timesteps, no_of_predictors))
    print("The input shape is: {}".format(X_train.shape))
    return X_train

# Building the RNN 

''' FUNCTION PARAMETERS - (data,  
num_timesteps: number of previous days used for the feedback loop,
batch_size: size of each batch,
input_dim: number of predictors'''

def build_RNN(num_layers = 4, num_timesteps, dropout_rate = 0.2, num_epochs = 100):
    # Building the regressor
    regressor = Sequential()

    # Adding the first LSTM layer and a dropout regularisation layer
    regressor.add(LSTM(units = num_timesteps, 
                    return_sequences = True, 
                    input_shape = (X_train.shape[1], 1)))

    # Dropout regularisation 
    regressor.add(Dropout(rate = dropout_rate)) # 20% dropout

    # Adding the hidden layers
    for i in range(num_layers - 1): 
    	# Adding an input layer
    	regressor.add(LSTM(units = num_timesteps, return_sequences = True))
    	regressor.add(Dropout(rate = dropout_rate))

    # Adding the output layer
    regressor.add(Dense(units = 1))

    # Compiling our RNN 
    regressor.compile(optimizer = 'adam', 
                      loss = 'mean_squared_error')

    # Fitting the RNN to the Training Set
    history = regressor.fit(X_train, y_train, 
                            epochs = num_epochs, 
                            batch_size = 32, 
                            validation_split=0.25)
