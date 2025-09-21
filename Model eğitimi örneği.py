from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd

df = pd.read_csv(r'c:\users\alper\downloads\Student_Grades.csv') #readed csv file

y = df['Scores']                        #scores column in csv file
x = df.drop(columns=['Scores','Grade']) #Dropped the Scores and Grade column

y = y.values.astype('float32')          #converted dtypes(Data Types)
x = x.values.astype('float32')          #converted dtypes(Data Types)

model = Sequential()                    #Keras ai model

model.add(Dense(16))                    #input-layer
model.add(Dense(32,activation='relu'))  #hidden-layer
model.add(Dense(32,activation='relu'))  #hidden-layer
model.add(Dense(1))                     #output-layer

model.compile(optimizer='adam',loss='mse')
model.fit(x,y,epochs=128,batch_size=16)


#Burada Keras kütüphanesinin hazır modelini eğitmeye çalıştık 
#BU sadece örnek bir koddur
