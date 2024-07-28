import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import train_test_split, cross_val_predict, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from math import sqrt

from keras_tuner.tuners import RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Dense, Flatten, Activation
import time
from sklearn.metrics import r2_score
import keras_tuner as kt

from collections import Counter 
import pickle
import pandas as pd
import collections
import seaborn as sns
import tensorflow as tf
data = pd.read_csv('/Visc_ABCD/Visc_ABCD_New.csv')
X = pd.read_csv('/Visc_ABCD/Visc_AlvaDesc_ABCD.csv')
df = data
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Concatenate
from numpy.random import seed
import tensorflow
GS_result = pd.DataFrame(columns=['YM_Train R^2', 'YM_Test R^2','Network', 'test_size', 'epochs'])
Y = df[['Normalized_D']]
for num_state in range(1, 51):
        for num_batch in [1, 2, 4, 8,16,32,64]:
                xtrain, xtest, ytrain, ytest=train_test_split(X.values, Y.values, test_size=0.1, random_state=num_state)

                num_neural_1 = 32
                num_neural_2 = 1042
                num_neural_3 = 256
                num_neural_4 = 512


                A1 = Input(shape=(702),name='A1')
                A2 = Dense(num_neural_1, activation='relu',name='A2')(A1)
                A3 = Dense(num_neural_2, activation='relu',name='A3')(A2)
                A4 = Dense(num_neural_3, activation='relu',name='A4')(A3)
                A5 = Dense(num_neural_4, activation='relu',name='A5')(A4)
                A6 = Dense(1, name='A6')(A5)

                B2 = Dense(num_neural_1, activation='relu',name='B2')(A1)
                B3 = Dense(num_neural_2, activation='relu',name='B3')(B2)
                B4 = Dense(num_neural_3, activation='relu',name='B4')(B3)
                B5 = Dense(num_neural_4, activation='relu',name='B5')(B4)
                B6 = Dense(1, name='B6')(B5)

                C2 = Dense(num_neural_1, activation='relu',name='C2')(A1)
                C3 = Dense(num_neural_2, activation='relu',name='C3')(C2)
                C4 = Dense(num_neural_3, activation='relu',name='C4')(C3)
                C5 = Dense(num_neural_4, activation='relu',name='C5')(C4)
                C6 = Dense(1, name='C6')(C5)

                concat_layer = Concatenate()([A6, B6, C6])

                model = Model(inputs=[A1],outputs=concat_layer)
                model.compile(loss = "mse", optimizer = 'adam')

                seed(1)
                tensorflow.random.set_seed(1*7+333)
                history = model.fit(xtrain, np.hstack([ytrain, ytrain, ytrain]), epochs=100, batch_size=num_batch, validation_data = ((xtest), np.hstack([ytest, ytest, ytest])), verbose=0)
                y_pred_train = model.predict((xtrain))
                y_pred_train_reshaped = y_pred_train.reshape((5456, 3, 1))
                y_pred_train = y_pred_train_reshaped.mean(axis=1)
                YSTrR = r2_score(ytrain, y_pred_train)
                y_pred_test = model.predict((xtest))
                y_pred_test_reshaped =y_pred_test.reshape((607, 3, 1))
                y_pred_test = y_pred_test_reshaped.mean(axis=1)
                YSTeR = r2_score(ytest, y_pred_test)
                GS_result.loc[len(GS_result.index)]=[YSTrR,YSTeR,str(num_state)+"_"+str(num_batch),0.1,100]
                
                history = model.fit(xtrain, np.hstack([ytrain, ytrain, ytrain]), epochs=100, batch_size=num_batch, validation_data = ((xtest), np.hstack([ytest, ytest, ytest])), verbose=0)
                y_pred_train = model.predict((xtrain))
                y_pred_train_reshaped = y_pred_train.reshape((5456, 3, 1))
                y_pred_train = y_pred_train_reshaped.mean(axis=1)
                YSTrR = r2_score(ytrain, y_pred_train)
                y_pred_test = model.predict((xtest))
                y_pred_test_reshaped =y_pred_test.reshape((607, 3, 1))
                y_pred_test = y_pred_test_reshaped.mean(axis=1)
                YSTeR = r2_score(ytest, y_pred_test)
                GS_result.loc[len(GS_result.index)]=[YSTrR,YSTeR,str(num_state)+"_"+str(num_batch),0.1,200]

                GS_result.to_csv('result_Visc_logD_New_2(1).csv')
