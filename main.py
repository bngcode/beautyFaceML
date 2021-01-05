import os

from kerastuner import RandomSearch

import methods
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
BS = 64
#training_set=[]
#training_set = methods.loadImages(IMG_SIZE)
#random.shuffle(training_set)
#methods.pickle_it(training_set, IMG_SIZE)


#Load preprocessed data
X = pickle.load(open("X.pickle", "rb"))
Y = pickle.load(open("Y.pickle", "rb"))

#Split data, note that a test data set is missing. I have not uploaded my test data set as it contains private photos.
#Feel free to split the data further to create a test data set
n = len(X)
x_train = X[:int(n*0.8)]
y_train = Y[:int(n*0.8)]

x_val = X[int(n*0.8):]
y_val = Y[int(n*0.8):]

#Date augmention, note that normalizing data to [0,1] interval was already done in loadImages()
train_generator = ImageDataGenerator(rotation_range=4,
                                     brightness_range=(0.8, 1.2),
                                     horizontal_flip=True)

valid_generator = ImageDataGenerator(horizontal_flip=True)

train_iterator = train_generator.flow(x=x_train, y=y_train, batch_size=BS)
valid_iterator = valid_generator.flow(x=x_val, y=y_val, batch_size=BS)

train_steps = train_iterator.n // BS
valid_steps = valid_iterator.n // BS

#RandomSearch
MAX_TRIALS = 100
EXECUTION_PER_TRIAL = 3

tuner = RandomSearch(methods.build_model, objective='val_loss', max_trials=MAX_TRIALS, executions_per_trial=EXECUTION_PER_TRIAL,
    directory='random_search', project_name='beautyFaceDetection'
                     )
tuner.search(train_iterator, epochs=EPOCHS, validation_data=valid_iterator, steps_per_epoch=train_steps, validation_steps=valid_steps,)


#Make prediction with trained model
model = tf.keras.models.load_model("savedModel")
image = os.path.join(os.getcwd(), 'data\\otherImages\\beautifulWomen.jpg')
methods.predictImage(image, model, Y)
