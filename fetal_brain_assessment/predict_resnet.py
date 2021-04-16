#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import numpy as np
import pandas as pd
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

	def predict(self, stacked_data, row_names) -> pd.DataFrame:
		stacked_data = np.array(stacked_data, dtype=np.float32)

		# Normalize dataset
		min1 = np.amin(stacked_data)
		# max1 = np.amax(volumes)
		max1 = 10000
		logger.info('Min: %s', min1)
		logger.info('Max: %s', max1)
		stacked_data = (stacked_data - min1) / (max1 - min1)
		min1 = np.amin(stacked_data)
		max1 = np.amax(stacked_data)
		logger.info('New min: %s', min1)
		logger.info('New max: %s', max1)

		logger.debug('Doing prediction')
		prediction = self.model.predict(stacked_data, verbose=1 if logger.level < 25 else 0)

		df = pd.DataFrame(row_names, columns=['filename'])
		df['quality'] = prediction
		return df
