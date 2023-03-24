#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
from time import sleep
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from preprocess import preprocess
import keras
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import batch_normalization
from keras.layers.activation import LeakyReLU


# In[2]:


dataset = 0

# load the dataset
dataset = str(dataset)
data_path = "train_preprocessed/"+dataset+".npy"
if os.path.exists(data_path):
    # print(data_path+" exists.")
    train_data = np.load(data_path)
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], 1)
    train_labels = np.load(data_path[:-4]+"_labels.npy")
    print(data_path+" has been loaded successfully.")
else:
    print(data_path+" does not exist.")
    preprocess(dataset)


# In[3]:


train_labels_one_hot = to_categorical(train_labels)
# train_labels_one_hot[20]
train_X, test_X, train_label, test_label = train_test_split(train_data, train_labels_one_hot, test_size=0.2, random_state=13)
train_X.shape, test_X.shape, train_label.shape, test_label.shape


# In[4]:


batch_size = 64
epochs = 20
num_classes = len(np.unique(train_labels))


# In[5]:


model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), activation='linear', input_shape=train_X.shape[1:], padding="same"))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2,2), padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3), activation='linear', padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3), activation='linear', padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128,activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation="softmax"))


# In[6]:


model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
# model.summary()


# In[ ]:


model_train = model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(test_X, test_label))


# In[ ]:


model.save("model.h5py")

