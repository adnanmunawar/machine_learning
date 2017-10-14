from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout
from keras.datasets import mnist
import keras.activations
from keras.optimizers import SGD
from keras.metrics import categorical_accuracy
import numpy as np

import gym


model_name = 'CartPole-v0'

model = gym.make(model_name)
var =  model.action_space
print var
model.reset()
action = [0,1]
for i in range(500):
    model.render()
    model.step(action[i%2])