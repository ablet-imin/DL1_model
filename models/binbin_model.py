import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense, Activation
import tensorflow.keras as keras
from tensorflow.keras import backend as K

class Dropout(tf.keras.layers.Dropout):
	"""Applies Dropout to the input.
	Dropout consists in randomly setting
	a fraction `rate` of input units to 0 at each update during training time,
	which helps prevent overfitting.
	# Arguments
	    rate: float between 0 and 1. Fraction of the input units to drop.
	    noise_shape: 1D integer tensor representing the shape of the
	        binary dropout mask that will be multiplied with the input.
		For instance, if your inputs have shape
		`(batch_size, timesteps, features)` and
		you want the dropout mask to be the same for all timesteps,
		you can use `noise_shape=(batch_size, 1, features)`.
	    seed: A Python integer to use as random seed.
	# References
	    - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](
	       http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
	"""
	def __init__(self, rate, training=None, noise_shape=None, seed=None, **kwargs):
		super(Dropout, self).__init__(rate, noise_shape=None, seed=None,**kwargs)
		self.training = training

	def call(self, inputs, training=None):
		if 0. < self.rate < 1.:
			noise_shape = self._get_noise_shape(inputs)

			def dropped_inputs():
				return K.dropout(inputs, self.rate, noise_shape,seed=self.seed)

			if not training:
				return K.in_train_phase(dropped_inputs, inputs, training=self.training)

			return K.in_train_phase(dropped_inputs, inputs, training=training)

		return inputs

def DL1_model(InputShape, training):
	"""have some doubts about InputShape"""
	model = Sequential()
	model.add(Dense(units=78, input_shape=[InputShape,],
			activation='relu',
			kernel_initializer='glorot_uniform'
			))
	model.add(Activation('linear'))
	model.add(Dropout(rate=0.1, training=training))
	model.add(BatchNormalization())
	# -------------------------- 2nd layer -----------------------#
	model.add(Dense(units=66,
		activation='relu',
		kernel_initializer='glorot_uniform',
		))
	model.add(Dropout(rate=0.2, training=training))
	model.add(BatchNormalization())
	# -------------------------- 3rd layer -----------------------#
	model.add(Dense(units=57,
		activation='relu',
		kernel_initializer='glorot_uniform',
		))
	model.add(Dropout(rate=0.2, training=training))
	model.add(BatchNormalization())
	# -------------------------- 4th layer -----------------------#
	model.add(Dense(units=48,
		activation='relu',
		kernel_initializer='glorot_uniform',
		))
	model.add(Dropout(rate=0.2, training=training))
	model.add(BatchNormalization())
	# -------------------------- 5th layer -----------------------#
	model.add(Dense(units=36,
		activation='relu',
		kernel_initializer='glorot_uniform',
		))
	model.add(Dropout(rate=0.2, training=training))
	model.add(BatchNormalization())
	# -------------------------- 6th layer -----------------------#
	model.add(Dense(units=24,
		activation='relu',
		kernel_initializer='glorot_uniform',
		))
	model.add(Dropout(rate=0.2, training=training))
	model.add(BatchNormalization())
	# -------------------------- 7th layer -----------------------#
	model.add(Dense(units=12,
		activation='relu',
		kernel_initializer='glorot_uniform',
		))
	model.add(Dropout(rate=0.2, training=training))
	model.add(BatchNormalization())
	# -------------------------- 8th layer -----------------------#
	model.add(Dense(units=6,
		activation='relu',
		kernel_initializer='glorot_uniform',
		))
	model.add(Dropout(rate=0.2, training=training))
	model.add(BatchNormalization())
	# -------------------------- 9th layer -----------------------#
	model.add(Dense(3, input_shape=(6,),
		activation="softmax", kernel_initializer='glorot_uniform'))

	#model.summary()

	model_optimizer = Adam(lr=0.0005)

	model.compile( # loss='mse',
			loss = 'categorical_crossentropy',
			optimizer=model_optimizer,
			metrics=['accuracy']
			)

	return model
