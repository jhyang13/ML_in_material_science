# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 11:32:45 2021

@author: Administrator
"""
import numpy as np
import sklearn
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dropout, Dense, GRU, LSTM
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

input = np.loadtxt('b25_gausse.umat')
output = np.loadtxt('kirchhoff25_gausse.umat')
#strain_stress_e = abs(strain_stress_e)
#strain_stress_e = strain_stress_e**2
input=np.reshape(input[:,0],(50000,1))
output=np.reshape(output[:,0],(50000,1))
min_max=np.zeros(2*(input.shape[1]+output.shape[1]))
for i in range(input.shape[1]):
    min_max[2*i]=input[:,i].max()
    min_max[2*i+1] = input[:,i].min()
for i in range(output.shape[1]):
    j=i+input.shape[1]
    min_max[2*j]=output[:,i].max()
    min_max[2*j+1] = output[:,i].min()

sc = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))  # 定义归一化：归一化到(0，1)之间
input_scale = sc.fit_transform(input)
output_scale = sc.fit_transform(output)

X = np.reshape(input_scale, (100,500, 1))
Y = np.reshape(output_scale, (100,500, 1))

x_train, x_test, y_train, y_test = model_selection.\
    train_test_split(X, Y, test_size=0.15, random_state=1)



model = tf.keras.Sequential([
#    LSTM( 2, return_sequences=True, activation='relu'),
#    GRU( 20, return_sequences=True,activation='tanh', recurrent_activation='sigmoid'),
#    LSTM( 128, return_sequences=True,activation='tanh', recurrent_activation='sigmoid'),

    GRU(256, return_sequences=True, activation='tanh',
        recurrent_activation='sigmoid'),
    GRU(256, return_sequences=True, activation='tanh',
        recurrent_activation='sigmoid'),
    GRU(128, return_sequences=True, activation='tanh',
        recurrent_activation='sigmoid'),
#    GRU(128, return_sequences=True, activation='tanh',
#        recurrent_activation='sigmoid'),
    # LSTM( 100, return_sequences=True,activation='tanh', recurrent_activation='sigmoid'),
    # Dense(8, activation='tanh', use_bias=True),
    # Dense(8, activation='tanh', use_bias=True),
    # Dense(8, activation='tanh', use_bias=True),
    Dense(8,activation='tanh'),
    # Dense(8,activation='tanh'),
    Dense(2,activation='linear')
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005,clipvalue=0.5,decay=0.001),
              loss='mean_squared_error')  # 损失函数用均方误差 Adam ,decay=0.00015,learning_rate=0.01
# 该应用只观测loss数值，不观测准确率，所以删去metrics选项，一会在每个epoch迭代显示时只显示loss值

checkpoint_save_path = "./checkpoint2/stock.ckpt"

if False:
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)
 
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss')


history = model.fit(x_train, y_train, batch_size=48, epochs=2500, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=cp_callback, use_multiprocessing=True)

model.summary()
model.save("YTL-RNN")
loss = history.history['loss']
val_loss = history.history['val_loss']


#plt.ylim([0,1])
plt.semilogy(loss, label='Training Loss')
plt.semilogy(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
plt.savefig("loss_fig.jpg")
plt.savefig("loss_fig.png")

file = open('weights.txt', 'w')  # 参数提取
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

##############################################################################
#########################export to fortan code################################
netinfo2 = [len(model.layers)]
netinfo2.append(x_train.shape[2])
netinfo2.append(model.layers[len(model.layers)-1].units)
for ll in model.layers:
    netinfo2.append(ll.units)
    if ll.name.find('lstm')!=-1:
        netinfo2.append(1)
    elif ll.name.find('dense')!=-1:
        netinfo2.append(0)
    elif ll.name.find('gru')!=-1:
        netinfo2.append(2)
    else:
        print('error: no layer type')
        netinfo2.append(999)
    if str(ll.activation).find('relu')!=-1:
        netinfo2.append(1)
    elif str(ll.activation).find('tanh')!=-1:
        netinfo2.append(2)
    elif str(ll.activation).find('sigmoid')!=-1:
        netinfo2.append(3)
    elif str(ll.activation).find('linear')!=-1:
        netinfo2.append(4)
    else:
        print('error: no active type') 
        netinfo2.append(999)
    if ll.name.find('dense')!=-1:
        netinfo2.append(99999)
    else:
        if str(ll.recurrent_activation).find('relu')!=-1:
            netinfo2.append(1)
        elif str(ll.recurrent_activation).find('tanh')!=-1:
            netinfo2.append(2)
        elif str(ll.recurrent_activation).find('sigmoid')!=-1:
            netinfo2.append(3)
        elif str(ll.recurrent_activation).find('linear')!=-1:
            netinfo2.append(4)
        else:
            print('error: no active type') 
            netinfo2.append(999)
file = open('./weights2.txt', 'w')
for data in netinfo2:
    file.write('%d\n'%data)
for v in min_max:
    file.write('%18.9e\n' % v)
for v in model.trainable_variables:
    variable = v.numpy().reshape(-1,1)
    for data in variable:
        file.write('%35.30e\n'%data)
file.close()



