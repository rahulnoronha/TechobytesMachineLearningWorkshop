
import numpy as np
import pandas as pd
#from numpy import genfromtxt
import matplotlib.pyplot as plt
plt.ion()

data = pd.read_csv('my_data.csv',sep=';')
noise1 = np.random.rand(400,1) 
noise2 = np.random.rand(400,1)
noise3 = np.random.rand(400,1)

dependent = data[['price']] + 20*noise1 - 30*noise2 + 40*noise3
independent = data[['value1','value2']] 

#plt.plot(idx,values,'o')
#plt.plot(variables[:1],cost,'o')

#plt.scatter(variables[:,0],variables[:,1],s=cost*100)

import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(1,input_shape=(2,)))
#model.add(Dense(1))
#model.add(Activation('linear'))


model.compile(keras.optimizers.Adam(lr=0.8),'mean_squared_error')

H = model.fit(independent,dependent,epochs=20,batch_size=10)

predicted_values = model.predict(independent)

plt.plot(dependent[0:100],'r*')
plt.plot(predicted_values[0:100],'b-')
#plt.plot(H.history['loss'])
