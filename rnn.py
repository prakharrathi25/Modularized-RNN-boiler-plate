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

# Error Evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

''' FUNCTION PARAMETERS - (num_layers: Number of layers needed in the neural network including the input layer and excluding the output layer,  
num_timesteps: number of previous days used for the feedback loop,
dropout_rate: the percentage of data you need to drop for this regularisation,
num_epochs: number of epochs for each batch'''

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

# Plotting the RNN history 
def print_rnn_history(history, validation = False):
    print(history.history.keys())
    if validation == True: 
        plt.plot(history.history['loss'], label = 'Loss')
        plt.plot(history.history['val_loss'], label = 'Val_loss')
    else: 
        plt.plot(history.history['loss'], label = 'Loss')

# Making the predictions on the test data
def make_pred(data=feature_data): 

	# Acquiring the dataset to make predictions on
	actual_values = test_set
	print("The size of the test set is: {}".format(len(real_index_values)))

	# Prediction inputs
	inputs = data[len(data) - len(test_set) - 60:]

	# Reshaping the input array 
	inputs = inputs.reshape(-1, 1)

	# Scaling the input data
	inputs = scaler.transform(inputs)

	# Prediction Sets
	X_test = []
	for i in range(60, 1251): 
	    X_test.append(inputs[i - 60 : i, 0])
	X_test = np.array(X_test)
	X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
	print("The length of the dataset for predictions: {}".format(len(X_test)))

	# Making predictions
	predicted_index = regressor.predict(X_test)
	predicted_index = scaler.inverse_transform(predicted_index)
	return predicted_index, actual_index

# Measuring the Error
''' FUNCTION PARAMETERS - (predicted_index: the values of the predicted data
actual_index: the actual test values
'''
def error_measure(predicted_index, actual_index):
	mae_acc = mean_absolute_error(y_true = actual_index, y_pred = predicted_index)
	mse_acc = mean_squared_error(y_true = actual_index, y_pred = predicted_index)
	r2_score = r2_score(y_true = actual_index, y_pred = predicted_index)

	print("Mean Absolute Error: {}".format(mae_acc))
	print("Mean Squared Error: {}".format(mse_acc))
	print("R2 Score: {}".format(r2_score))

# Visualisation of the results
def plot_values(predicted_index, actual_index):
	plt.figure(figsize=(15,6))
	plt.plot(predicted_index, color = 'r', label = 'Predicted Values')
	plt.plot(actual_index, color = 'g', label = 'Actual Values')
	plt.title('RSI future predictions')

	plt.xlabel('Time')
	plt.ylabel('RSI')
	plt.legend()
	plt.show()