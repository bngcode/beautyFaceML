import os
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
training_set=[]
training_set = methods.loadImages(IMG_SIZE)
random.shuffle(training_set)
methods.pickle_it(training_set, IMG_SIZE)


#Load preprocessed data
X = pickle.load(open("X.pickle", "rb"))
Y = pickle.load(open("Y.pickle", "rb"))


# define the grid search parameters, feel free to edit
batch_size = [64, 128, 128*2]
epochsGrid = [300]
dropoutGrid = [0.5, 0.25]
learning_rate = [0.01, 0.001, 0.0001]
size_histories = {}

min_val_loss = 10000
best_para = {}

#Grid search, fine tuning CNN
for dr in dropoutGrid:
  for epochs in epochsGrid:
    for batch in batch_size:
      for lr in learning_rate:
        model = methods.create_model(IMG_SIZE, lr, dr)
        model_name = str(epochs) + '_' + str(batch) + '_' + str(lr)
        my_callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto',
    baseline=None, restore_best_weights=True)]
        size_histories[model_name] = model.fit(X, Y, batch_size=batch, epochs=epochs, validation_split=0.1, callbacks=my_callbacks, verbose=2)
        if min(size_histories[model_name].history['val_loss']) < min_val_loss:
          min_val_loss = min(size_histories[model_name].history['val_loss'])
          best_para['epoch'] = epochs
          best_para['batch'] = batch
          best_para['lr'] = lr
          model.save('savedModel')





#Make prediction with trained model
model = tf.keras.models.load_model("savedModel")
image = os.path.join(os.getcwd(), 'data\\otherImages\\beautifulWomen.jpg')
methods.predictImage(image, model, Y)
