import os
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers
from tensorflow import keras
from keras.layers import Dropout

import matplotlib.pyplot as plt
import pickle
import numpy as np
import cv2


def create_model(IMG_SIZE, lr, dr):
  #Limit memore usage of GPU
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      tf.config.experimental.set_virtual_device_configuration(gpus[0], [
        tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*7)])
    except RuntimeError as e:
      print(e)

  model = keras.Sequential()
  model.add(MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False))
  model.layers[0].trainable = False
  model.add(layers.GlobalAveragePooling2D())
  model.add(layers.BatchNormalization())
  model.add(layers.Dense(256*5, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
  model.add(Dropout(dr))
  model.add(layers.Dense(1, activation=tf.keras.layers.LeakyReLU(alpha=0.3)))
  adam = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.98,
                                       epsilon=1e-9)
  sgd = tf.keras.optimizers.SGD(lr=lr, decay=1e-6, momentum=0.5)
  model.compile(optimizer=adam,
                loss=tf.keras.losses.mean_squared_error)

  model.summary()
  return model


def loadImages(IMG_SIZE):
  path = os.path.join(os.getcwd(), 'data\\Images')
  training_data=[]
  labelMap = getLabelMap()
  for img in os.listdir(path):
    out_array = np.zeros((350,350, 3), np.float32)
    try:
      img_array = cv2.imread(os.path.join(path, img))
      img_array=img_array.astype('float32')
      out_array = cv2.normalize(img_array, out_array, 0, 1, cv2.NORM_MINMAX)
      out_array = cv2.resize(out_array, (IMG_SIZE, IMG_SIZE))

      #image = cv2.cvtColor(out_array, cv2.COLOR_BGR2RGB)
      #plt.imshow(image)
      #plt.show()

      training_data.append([out_array, float(labelMap[img])])
    except Exception as e:
      pass
  return training_data

def getLabelMap():
  map = {}
  path = os.getcwd()
  path = os.path.join(path, "data\\train_test_files\\All_labels.txt")
  f = open(path, "r")
  for line in f:
    line = line.split()
    map[line[0]] = line[1]
  f.close()
  return map


def showimg(image):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  plt.imshow(image)
  plt.show()


def pickle_it(training_set, IMG_SIZE):
  X = []
  Y = []
  for features, label in training_set:
    X.append(features)
    Y.append(label)
  X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
  Y = np.array(Y)

  pickle_out = open("X.pickle", "wb")
  pickle.dump(X, pickle_out)
  pickle_out.close()

  pickle_out = open("Y.pickle", "wb")
  pickle.dump(Y, pickle_out)
  pickle_out.close()


def betterThan(y, Y):
  Z=np.sort(Y)
  cnt = 0
  for z in Z:
    if z>y:
      break
    else:
      cnt = cnt+1
  return float(cnt/len(Y))

def predictImage(image, model, Y):
  out_array = np.zeros((IMG_SIZE,IMG_SIZE,3))
  img_array = cv2.imread(image)
  img_array = img_array.astype('float32')
  out_array = cv2.normalize(img_array, out_array, 0, 1, cv2.NORM_MINMAX)
  out_array = cv2.resize(out_array, (IMG_SIZE, IMG_SIZE))
  out_array =  np.array(out_array).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
  y = model.predict(out_array)
  per = betterThan(y, Y)
  print('You look better than ' + str(per) + '% of the dataset')
