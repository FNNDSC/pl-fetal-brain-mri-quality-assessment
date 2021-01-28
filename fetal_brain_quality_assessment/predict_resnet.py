#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Developed by: Ivan Legorreta
Contact information: ilegorreta@outlook.com
'''

import nibabel as nib
import os
import numpy as np
import pandas as pd
import h5py
import time
import csv
import sys
import fnmatch
from tensorflow import keras
import tensorflow as tf
from keras.utils import to_categorical
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, MaxPooling3D, GlobalAveragePooling3D, Add
from keras.layers import Dropout, Input, BatchNormalization, Activation
#from sklearn.metrics import confusion_matrix, accuracy_score
from keras.losses import mean_squared_error, huber_loss
from keras.optimizers import Adadelta, SGD, Adam
from keras.models import Model, Sequential
from keras.utils import to_categorical
from keras.utils.io_utils import HDF5Matrix
import matplotlib.pyplot as plt
#from sklearn.utils import shuffle
#from generator import NumpyArrayIterator
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold, KFold
from keras import backend as K
started_at = time.asctime()
from scipy.stats import spearmanr

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'

#assert tf.test.is_gpu_available()
#assert tf.test.is_built_with_cuda()

# Define the Huber loss so that it can be used with Keras
def huber_loss_wrapper(**huber_loss_kwargs):
    def huber_loss_wrapped_function(y_true, y_pred):
        return huber_loss(y_true, y_pred, **huber_loss_kwargs)
    return huber_loss_wrapped_function

#Identity Block for ResNet
def id_block(X, f):
	
	X_shortcut = X

	#First component of main path
	X = BatchNormalization() (X)
	X = Activation('relu')(X)
	X = Conv3D(filters=int(f/4), kernel_size=1, kernel_initializer='he_uniform', padding='same') (X)

	#Second component of main path
	X = BatchNormalization() (X)
	X = Activation('relu')(X)
	X = Conv3D(filters=int(f/4), kernel_size=3, kernel_initializer='he_uniform', padding='same') (X)

	#Third component of main path
	X = BatchNormalization() (X)
	X = Activation('relu')(X)
	X = Conv3D(filters=f, kernel_size=1, kernel_initializer='he_uniform', padding='same') (X)

	#Merge paths with skip connection
	X = Add()([X, X_shortcut])

	return X

dataDir = sys.argv[1]

#Counting the number of crop images to predict
count = len(fnmatch.filter(os.listdir(dataDir), '*_crop.nii'))

#Important to change the first axis depending on the number of volumes
volumes = np.zeros([count,217,178,60,1], dtype='float32')
print(volumes.dtype)
names = list()

cont = 0
print("Begining...")
for img in sorted(os.listdir(dataDir)):
	if img.endswith("_crop.nii"):
		#print(img)
		names.append(img)
		example_filename = os.path.join(dataDir, img)
		image = nib.load(example_filename)
		data = image.get_fdata()
		data = np.float32(data)
		print(img, data.shape)
		#if data.shape > (217, 178, 60):
		#	continue
		data = np.nan_to_num(data)
		data[data < 0] = 0
		data[data >= 10000] = 10000
		data = np.expand_dims(data, axis=3)
		pad = np.zeros([217,178,60,1], dtype='float32')
		pad[:data.shape[0],:data.shape[1],:data.shape[2]] = data
		volumes[cont] = pad
		cont = cont + 1

#Normalizing dataset
min1 = np.amin(volumes)
#max1 = np.amax(volumes)
max1 = 10000
print('Min: ',min1)
print('Max: ',max1)
volumes = (volumes - min1) / (max1 - min1)
min1 = np.amin(volumes)
max1 = np.amax(volumes)
print('New Min1: ',min1)
print('New Max1: ',max1)

input_imgs = Input(shape=(217, 178, 60, 1))

conv2 = Conv3D(filters=2, kernel_size=(3, 3, 3), kernel_initializer='he_uniform', padding='same') (input_imgs)

maxPool1 = MaxPool3D(pool_size=(2, 2, 2), strides=(2,2,2)) (conv2)

bn4 = BatchNormalization() (maxPool1)
act4 = Activation('relu') (bn4)
conv4 = Conv3D(filters=4, kernel_size=(3, 3, 3), kernel_initializer='he_uniform', padding='same') (act4)

maxPool2 = MaxPool3D(pool_size=(2, 2, 2), strides=(2,2,2)) (conv4)

bn8 = BatchNormalization() (maxPool2)
act8 = Activation('relu') (bn8)
conv8 = Conv3D(filters=8, kernel_size=(3, 3, 3), kernel_initializer='he_uniform', padding='same') (act8)

bn16 = BatchNormalization() (conv8)
act16 = Activation('relu') (bn16)
conv16 = Conv3D(filters=16, kernel_size=(3, 3, 3), kernel_initializer='he_uniform', padding='same') (act16)

conv_block16 = id_block(conv16, 16)
conv_block16 = id_block(conv_block16, 16)
conv_block16 = id_block(conv_block16, 16)
conv_block16 = id_block(conv_block16, 16)
conv_block16 = id_block(conv_block16, 16)
conv_block16 = id_block(conv_block16, 16)
conv_block16 = id_block(conv_block16, 16)
conv_block16 = id_block(conv_block16, 16)

maxPool3 = MaxPool3D(pool_size=(2, 2, 2), strides=(2,2,2)) (conv_block16)

bn32 = BatchNormalization() (maxPool3)
act32 = Activation('relu') (bn32)
conv32 = Conv3D(filters=32, kernel_size=(3, 3, 3), kernel_initializer='he_uniform', padding='same') (act32)

bn64 = BatchNormalization() (conv32)
act64 = Activation('relu') (bn64)
conv64 = Conv3D(filters=64, kernel_size=(3, 3, 3), kernel_initializer='he_uniform', padding='same') (act64)	

maxPool4 = MaxPool3D(pool_size=(2, 2, 2), strides=(2,2,2)) (conv64)

bn128 = BatchNormalization() (maxPool4)
act128 = Activation('relu') (bn128)
conv128 = Conv3D(filters=128, kernel_size=(3, 3, 3), kernel_initializer='he_uniform', padding='same') (act128)

bn256 = BatchNormalization() (conv128)
act256 = Activation('relu') (bn256)
conv256 = Conv3D(filters=256, kernel_size=(3, 3, 3), kernel_initializer='he_uniform', padding='same') (act256)

bn512 = BatchNormalization() (conv256)
act512 = Activation('relu') (bn512)
conv512 = Conv3D(filters=512, kernel_size=(3, 3, 3), kernel_initializer='he_uniform', padding='same') (act512)

conv_block512 = id_block(conv512, 512)
conv_block512 = id_block(conv_block512, 512)
conv_block512 = id_block(conv_block512, 512)
conv_block512 = id_block(conv_block512, 512)
conv_block512 = id_block(conv_block512, 512)
conv_block512 = id_block(conv_block512, 512)
conv_block512 = id_block(conv_block512, 512)
conv_block512 = id_block(conv_block512, 512)
conv_block512 = id_block(conv_block512, 512)
conv_block512 = id_block(conv_block512, 512)
conv_block512 = id_block(conv_block512, 512)

bn1024 = BatchNormalization() (conv_block512)
act1024 = Activation('relu') (bn1024)
conv1024 = Conv3D(filters=1024, kernel_size=(3, 3, 3), kernel_initializer='he_uniform', padding='same') (act1024)

conv_block1024 = id_block(conv1024, 1024)
conv_block1024 = id_block(conv_block1024, 1024)
conv_block1024 = id_block(conv_block1024, 1024)
conv_block1024 = id_block(conv_block1024, 1024)
conv_block1024 = id_block(conv_block1024, 1024)
conv_block1024 = id_block(conv_block1024, 1024)

bn2048 = BatchNormalization() (conv_block1024)
act2048 = Activation('relu') (bn2048)
conv2048 = Conv3D(filters=2048, kernel_size=(3, 3, 3), kernel_initializer='he_uniform', padding='same') (act2048)

conv_block2048 = id_block(conv2048, 2048)
conv_block2048 = id_block(conv_block2048, 2048)
conv_block2048 = id_block(conv_block2048, 2048)
conv_block2048 = id_block(conv_block2048, 2048)
conv_block2048 = id_block(conv_block2048, 2048)
conv_block2048 = id_block(conv_block2048, 2048)

conv_block2048 = BatchNormalization() (conv_block2048)
conv_block2048 = Activation('relu') (conv_block2048)

globalAP = GlobalAveragePooling3D() (conv_block2048)
dense1 = Dense(units=256, activation='relu', kernel_initializer='he_uniform') (globalAP)
dp = Dropout(0.4) (dense1)
dense2 = Dense(units=1, activation='linear') (dp)

model = Model(inputs=input_imgs, outputs=dense2)
print("Model created successfully!!")

#Compiling the model
model.compile(loss=huber_loss_wrapper(delta=0.15), optimizer=Adam(lr = 0.0001), metrics=['mean_absolute_error'])
print("Model compiled successfully!!")

#Creating dataset
pred_df = pd.DataFrame(names, columns=['Name'])

#Load weights of each fold
for fold in range(1,11):
	
	print('Predicting Fold %s' % fold)
	model.load_weights("/neuro/labs/grantlab/research/ivan_gonzalez/Documents/CNN/resnet/weights_resnet_sw2_k%s.hdf5" % fold)

	#Making predictions on the test set
	print('Predicting dataset:')
	prediction = model.predict(volumes, verbose=1)

	print('Saving folds predictions...')
	# Add Fold to a new column
	pred_df['QC Prediction Fold %s' %fold] = prediction

	#We only want 1 fold for now
	#break 

#Determine average value from all the folds
pred_df["Average"] = pred_df.mean(axis=1)

# Save results to CSV
print('Saving predictions to csv file...')
pred_df.to_csv('%s/predictions.csv' %dataDir, mode = 'a')

