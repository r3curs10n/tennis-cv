import numpy as np
np.random.seed(123)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Convolution2D, MaxPooling2D

from keras.utils import np_utils

import data_loader

((X_train, Y_train), (X_test, Y_test)) = data_loader.get_train_and_test_data()

width = X_train.shape[2]
height = X_train.shape[1]

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

model = Sequential()

model.add(Convolution2D(32, (5, 5), data_format='channels_first', activation='relu', input_shape=(1,height, width)))
model.add(MaxPooling2D(pool_size=(4,4)))

model.add(Convolution2D(64, (3, 3), data_format='channels_first', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=X_train.shape[0], epochs=20, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=0)
print score
model.save('models/court_detector.h5')

