
from keras.datasets import mnist
from matplotlib import pyplot as plt
from keras import utils as np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

import numpy as np
np.random.seed(123)  # for reproducibility


#from tensorflow.python.client import device_lib  
#print(device_lib.list_local_devices())

#load data 

(x_train,y_train),(x_test,y_test)= mnist.load_data()

#print(x_train.shape)
#plt.imshow(x_train[1])
#plt.show()

#reshaping data (60000,28, 28) ---> (60000, 1, 28, 28) to get the depth of the image
x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
#print(x_train.shape)

#Preprocess input data
x_train = x_train.astype('float32')
x_test = x_train.astype('float32')

x_train/=255
x_test/=255

#print(y_train.shape) #(60000,)
#print (y_train[:10])
#We have a 1D-array we need to convert to 10 classes each one for 1 digit 0 to 9


#Preprocess class labels
y_train = np_utils.to_categorical(y_train,10)
y_test = np_utils.to_categorical(y_test,10)

print (y_train.shape)  #60000 train data  , now we have 10 classes
print (y_test.shape)   #10000 test data

print (x_test.shape)
print (x_train.shape)

####### Creating  the model architecture #######

model = Sequential()
#inuput layer
model.add(Convolution2D(32, (3,3), activation='relu', input_shape=(1,28,28),data_format='channels_first'))      # 32 : Filters :the number of output filters in the convolution.
print ("model : ",model.output_shape)                                                                           # 3  : number o rows in each conv kernel (filter)
                                                                                                                # (1,28,28) stands for ( depth , width , height ) of a single input image
model.add(Convolution2D(32, (3,3), activation='relu',data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2,2),data_format="channels_first"))
model.add(Dropout(0.25))
print ("model1 : ",model.output_shape) 

#Flattening the model and creating the FC-layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

#we have 10 outputs 
model.add(Dense(10, activation='sigmoid'))
print ("model 2 : ",model.output_shape)

#Compiling the Model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#Fitting the data into the model, precising the batch size and epochs num

model.fit(x_train,y_train,batch_size=32, epochs=10, verbose=1)

#testing the model on test data 

classes = model.predict(x_test, batch_size=128)

score = model.evaluate(x_train,y_train,verbose=0)

print("\n%s: %.2f%%" % (model.metrics_names[1], score[1]*100))


