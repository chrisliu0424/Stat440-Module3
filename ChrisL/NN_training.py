import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import numpy as np
from tensorflow.keras.utils import to_categorical


X = pickle.load(open("X_train.pickle","rb"))
y = pickle.load(open("y_train.pickle","rb"))
y = np.array([int(string) for string in y])
y = to_categorical(y)

print((X.shape))
print(len(y))
print(X.shape)
X = X/255.0

model = Sequential()
model.add(Conv2D(64,(3,3),input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(7,activation='softmax'))

model.compile(loss = "categorical_crossentropy",
	optimizer="adam",
	metrics=['accuracy'])

model.fit(X,y,batch_size = 32,validation_split=0.1,epochs=5)

