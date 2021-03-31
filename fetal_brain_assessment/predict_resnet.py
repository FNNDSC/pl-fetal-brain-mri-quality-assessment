#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import basename
import logging
import numpy as np
import pandas as pd
import nibabel as nib
from keras.losses import mean_squared_error, huber_loss
from keras.optimizers import Adadelta, SGD, Adam

from fetal_brain_assessment.resnet_architecture import model_architecture as create_model_architecture

logger = logging.getLogger(__name__)


class Predictor:
	def __init__(self, weights='/usr/local/share/fetal_brain_assessment/weights_resnet.hdf5'):
		logger.debug('Creating model')
		self.model = create_model_architecture()
		logger.debug('Model created')

		self.model.compile(
			loss=lambda y_true, y_pred: huber_loss(y_true, y_pred, delta=0.15),
			optimizer=Adam(lr=0.0001),
			metrics=['mean_absolute_error'])
		logger.debug('Model compiled')
		logger.debug('Loading resnet weights from %s', weights)
		self.model.load_weights(weights)
		logger.debug('Predictor object setup complete.')

	@staticmethod
	def load_volume(filename) -> np.array:
		"""
		Load NIFTI volume as a numpy array. The data values are coerced into the dimensions
		(217, 178, 60) and values are restricted to between [0, 10000].
		"""
		logger.debug('Loading %s', filename)
		image = nib.load(filename)
		data = image.get_fdata()
		data = np.float32(data)
		if data.shape > (217, 178, 60):
			logger.warning('%s exceeds dimensions (217, 178, 60), skipped', filename)
		data = np.nan_to_num(data)
		data[data < 0] = 0
		data[data >= 10000] = 10000
		data = np.expand_dims(data, axis=3)
		pad = np.zeros([217, 178, 60, 1], dtype=np.float32)
		pad[:data.shape[0], :data.shape[1], :data.shape[2]] = data
		return pad

	def predict(self, input_files: list) -> pd.DataFrame:
		# stack raw data
		volumes = np.array([self.load_volume(f) for f in input_files], dtype=np.float32)

		# Normalize dataset
		min1 = np.amin(volumes)
		# max1 = np.amax(volumes)
		max1 = 10000
		logger.info('Min: %s', min1)
		logger.info('Max: %s', max1)
		volumes = (volumes - min1) / (max1 - min1)
		min1 = np.amin(volumes)
		max1 = np.amax(volumes)
		logger.info('New min: %s', min1)
		logger.info('New max: %s', max1)

		logger.debug('Doing prediction')
		prediction = self.model.predict(volumes, verbose=1 if logger.level < 25 else 0)

		df = pd.DataFrame(input_files, columns=['filename'])
		df['prediction'] = prediction
		return df
