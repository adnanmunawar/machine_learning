from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout
from keras.datasets import mnist
import keras.activations
from keras.optimizers import SGD
from keras.metrics import mean_absolute_percentage_error




model = Sequential()

[X_data, Y_data], [X_test, Y_test] = mnist.load_data()

X_data = X_data.reshape(X_data.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

Y_data = np_utils.to_categorical(Y_data, 10)
Y_test = np_utils.to_categorical(Y_test, 10)

print X_data.shape[1]
model.add(Dense(64, input_dim=X_data.shape[1], activation='sigmoid'))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr = 0.01, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['categorical_accuracy', 'acc'])

model.fit(X_data, Y_data, batch_size=64, epochs=5)

score = model.evaluate(X_test, Y_test, batch_size=64)
