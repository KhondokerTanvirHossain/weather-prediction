import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import model_from_json
#from temprature import loadTemparature
dataset_train = pd.read_csv('data.csv')
sc = MinMaxScaler(feature_range = (0, 1))

def lloadTemparature():
    dataset_train = pd.read_csv('data.csv')
    sc = MinMaxScaler(feature_range = (0, 1))
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    test_set = dataset_train.iloc[4001:4101, 2:3].values
    test_set_scaled = sc.fit_transform(test_set)
    test_set_reshaped = np.reshape(test_set_scaled, (test_set_scaled.shape[0], test_set_scaled.shape[1], 1))
    predicted_temprature = loaded_model.predict(test_set_reshaped)
    predicted_temprature = sc.inverse_transform(predicted_temprature)
    plt.plot(predicted_temprature, color = 'blue', label = 'Predicted Temparature')
    plt.plot(test_set, color = 'red', label = 'Real Temparature')
    plt.title('Temperature Prediction')
    plt.xlabel('Time')
    plt.ylabel('Temparute')
    plt.legend()
    plt.show()


lloadTemparature()
