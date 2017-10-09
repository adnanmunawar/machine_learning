from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout
from keras.datasets import mnist
import keras.activations
from keras.optimizers import SGD
from keras.metrics import categorical_accuracy
import numpy as np


def accuracy_percentage(y_true, y_pred):
    acc = 0
    tot = 0
    for i in range(0, y_true.shape[0]):
        if np.argmax(y_true[i,:]) == np.argmax(y_pred[i,:]):
            acc = acc + 1
        tot = tot+1
    percentage = acc/float(tot)
    print 'correct: ', acc, ', total: ', tot, ', percentage: ', percentage
    return percentage

pass


class Params():
    def __init__(self):
        pass
    batch_size = 64
    epoch = 5


def main():

    param = Params()
    model = Sequential()
    [X_data, Y_data], [X_test, Y_test] = mnist.load_data()
    X_data = X_data.reshape(X_data.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    Y_data = np_utils.to_categorical(Y_data, 10)
    Y_test = np_utils.to_categorical(Y_test, 10)
    print X_data.shape[1]
    model.add(Dense(32, input_dim=X_data.shape[1], activation='sigmoid'))
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(10, activation='softmax'))
    sgd = SGD(lr=0.01, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['categorical_accuracy', 'acc'])
    model.fit(X_data, Y_data, batch_size=param.batch_size, epochs=param.epoch, verbose=False)
    score = model.evaluate(X_test, Y_test, batch_size=param.batch_size, verbose=False)
    y_predictions = model.predict(X_test, batch_size=param.batch_size)
    print 'User computed accuracy is : ', accuracy_percentage(Y_test, y_predictions)
    print ' Score 1 is ', score


if __name__ == '__main__':
    main()