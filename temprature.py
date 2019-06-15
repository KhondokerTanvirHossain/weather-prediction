import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import model_from_json
import math
from sklearn.metrics import mean_squared_error
dataset_main = pd.read_csv('data.csv')
dataset = dataset_main.iloc[0:90000, 2:3].values
    
sc = MinMaxScaler(feature_range = (0, 1))
dataset_scaled = sc.fit_transform(dataset)


def trainTemprature():    
    #training_set = dataset.iloc[0:4001, 2:3].values
    #training_set_scaled = sc.fit_transform(training_set)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 5
    plt.rcParams["figure.figsize"] = fig_size
    
    plt.plot(dataset, color = 'blue', label = 'Temparature')
    plt.title('Temperature')
    plt.xlabel('Time')
    plt.ylabel('Temparute')
    plt.legend()
    plt.show()
    X_train = []
    y_train = []
    X_train = dataset_scaled[0:60000]
    y_train = dataset_scaled[1:60001]
    plt.plot(X_train, color = 'red', label = 'Scaled Temparature')
    plt.title('Scaled Temperature')
    plt.xlabel('Time')
    plt.ylabel('Temparute')
    plt.legend()
    plt.show()
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    regressor = Sequential()

    regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))
    regressor.add(Dropout(0.2))
    
    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

    regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)
    

    model_json = regressor.to_json()
    with open("modelTemp.json", "w") as json_file:
        json_file.write(model_json)
    regressor.save_weights("modelTemp.h5")
    print("Saved model to disk")


def loadTemparature():
    test_set = dataset_main.iloc[60001:62000, 2:3].values
    #test_set_scaled = sc.transform(test_set)
    test_set_scaled = dataset_scaled[60001:62000]
    json_file = open('modelTemp.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("modelTemp.h5")
    print("Loaded model from disk")
    test_set_reshaped = np.reshape(test_set_scaled, (test_set_scaled.shape[0], test_set_scaled.shape[1], 1))
    predicted_temprature = loaded_model.predict(test_set_reshaped)
    predicted_temprature = sc.inverse_transform(predicted_temprature)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 5
    plt.rcParams["figure.figsize"] = fig_size
    plt.plot(predicted_temprature, color = 'blue', label = 'Predicted Temparature')
    plt.plot(test_set, color = 'red', label = 'Real Temparature')
    plt.title('Temperature Prediction')
    plt.xlabel('Time')
    plt.ylabel('Temparute')
    plt.legend()
    plt.show()
    rmse = math.sqrt(mean_squared_error(test_set, predicted_temprature)) / 10
    print (rmse)
    
def predictionTemprature():
    #test_set = dataset_main.iloc[4001:4101, 2:3].values
    #test_set_scaled = sc.transform(test_set)
    test_set_scaled = dataset_scaled[80001:80865]
    json_file = open('modelTemp.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("modelTemp.h5")
    test_set_reshaped = np.reshape(test_set_scaled, (test_set_scaled.shape[0], test_set_scaled.shape[1], 1))
    predicted_temprature = loaded_model.predict(test_set_reshaped)
    predicted_temprature = sc.inverse_transform(predicted_temprature)
    #print(predicted_temprature)
    return predicted_temprature
