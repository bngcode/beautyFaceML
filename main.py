import os

from kerastuner import RandomSearch

import methods
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import pickle
import random


config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


#Preprocessing data
IMG_SIZE = 224
EPOCHS = 100
#training_set=[]
#training_set = methods.loadImages(IMG_SIZE)
#random.shuffle(training_set)
#methods.pickle_it(training_set, IMG_SIZE)


#Load preprocessed data
X = pickle.load(open("X.pickle", "rb"))
Y = pickle.load(open("Y.pickle", "rb"))

#Split data, note that a test data set is missing. I have not uploaded my test data set as it contains private photos. Feel free to split the data further to create a test data set
n = len(X)
x_train = X[:int(n*0.8)]
y_train = Y[:int(n*0.8)]

x_val = X[int(n*0.8):]
y_val = Y[int(n*0.8):]


#RandomSearch
MAX_TRIALS = 100
EXECUTION_PER_TRIAL = 3

tuner = RandomSearch(
    methods.build_model,
    objective='val_loss',
    max_trials=MAX_TRIALS,
    executions_per_trial=EXECUTION_PER_TRIAL,
    directory='random_search',
    project_name='beautyFaceDetection'
)


tuner.search(x_train, y_train,
             epochs=EPOCHS,
             validation_data=(x_val, y_val))


#Make prediction with trained model
model = tf.keras.models.load_model("savedModel")
image = os.path.join(os.getcwd(), 'data\\otherImages\\beautifulWomen.jpg')
methods.predictImage(image, model, Y)
