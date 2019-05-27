from keras import optimizers
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, Deconvolution2D
from keras.layers import Dropout, Flatten, BatchNormalization, LeakyReLU, Reshape
from keras.layers.merge import concatenate
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

import numpy as np


class SpeechEnhancementNetwork(object):

	def __init__(self, model):
		self.__model = model

	@classmethod
	def build(cls, audio_spectrogram_shape, video_shape):
		# append channels axis
		extended_audio_spectrogram_shape = list(audio_spectrogram_shape)
		extended_audio_spectrogram_shape.append(1)

		encoder, shared_embedding_size, audio_embedding_shape = cls.__build_encoder(
			extended_audio_spectrogram_shape, video_shape
		)

		decoder = cls.__build_decoder(shared_embedding_size, audio_embedding_shape)

		audio_input = Input(shape=extended_audio_spectrogram_shape)
		video_input = Input(shape=video_shape)

		audio_output = decoder(encoder([audio_input, video_input]))

		model = Model(inputs=[audio_input, video_input], outputs=audio_output)

		optimizer = optimizers.adam(lr=5e-4)
		model.compile(loss='mean_squared_error', optimizer=optimizer)

		model.summary()

		return SpeechEnhancementNetwork(model)

	@classmethod
	def __build_encoder(cls, extended_audio_spectrogram_shape, video_shape):
		audio_input = Input(shape=extended_audio_spectrogram_shape)
		video_input = Input(shape=video_shape)

		audio_embedding_matrix = cls.__build_audio_encoder(audio_input)
		audio_embedding = Flatten()(audio_embedding_matrix)

		video_embedding_matrix = cls.__build_video_encoder(video_input)
		video_embedding = Flatten()(video_embedding_matrix)

		x = concatenate([audio_embedding, video_embedding])
		shared_embedding_size = int(x._keras_shape[1] / 4)

		x = Dense(shared_embedding_size)(x)
		x = BatchNormalization()(x)
		shared_embedding = LeakyReLU()(x)

		model = Model(inputs=[audio_input, video_input], outputs=shared_embedding)
		model.summary()

		return model, shared_embedding_size, audio_embedding_matrix.shape[1:].as_list()

	@classmethod
	def __build_decoder(cls, shared_embedding_size, audio_embedding_shape):
		shared_embedding_input = Input(shape=(shared_embedding_size,))

		x = Dense(shared_embedding_size)(shared_embedding_input)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		audio_embedding_size = np.prod(audio_embedding_shape)

		x = Dense(audio_embedding_size)(x)
		x = Reshape(audio_embedding_shape)(x)
		x = BatchNormalization()(x)
		audio_embedding = LeakyReLU()(x)

		audio_output = cls.__build_audio_decoder(audio_embedding)

		model = Model(inputs=shared_embedding_input, outputs=audio_output)
		model.summary()

		return model

	@staticmethod
	def __build_audio_encoder(audio_input):
		x = Convolution2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same')(audio_input)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Convolution2D(64, kernel_size=(4, 4), strides=(1, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Convolution2D(128, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Convolution2D(128, kernel_size=(2, 2), strides=(2, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Convolution2D(128, kernel_size=(2, 2), strides=(2, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		return x

	@staticmethod
	def __build_audio_decoder(embedding):
		x = Deconvolution2D(128, kernel_size=(2, 2), strides=(2, 1), padding='same')(embedding)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Deconvolution2D(128, kernel_size=(2, 2), strides=(2, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Deconvolution2D(128, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Deconvolution2D(64, kernel_size=(4, 4), strides=(1, 1), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Deconvolution2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)

		x = Deconvolution2D(1, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)

		return x

	@staticmethod
	def __build_video_encoder(video_input):
		x = Convolution2D(128, kernel_size=(5, 5), padding='same')(video_input)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
		x = Dropout(0.25)(x)

		x = Convolution2D(128, kernel_size=(5, 5), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
		x = Dropout(0.25)(x)

		x = Convolution2D(256, kernel_size=(3, 3), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
		x = Dropout(0.25)(x)

		x = Convolution2D(256, kernel_size=(3, 3), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
		x = Dropout(0.25)(x)

		x = Convolution2D(512, kernel_size=(3, 3), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
		x = Dropout(0.25)(x)

		x = Convolution2D(512, kernel_size=(3, 3), padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU()(x)
		x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
		x = Dropout(0.25)(x)

		return x

	def train(self, train_mixed_spectrograms, train_video_samples, train_speech_spectrograms,
			  validation_mixed_spectrograms, validation_video_samples, validation_speech_spectrograms,
			  model_cache_path, tensorboard_dir):

		train_mixed_spectrograms = np.expand_dims(train_mixed_spectrograms, -1)  # append channels axis
		train_speech_spectrograms = np.expand_dims(train_speech_spectrograms, -1)  # append channels axis

		validation_mixed_spectrograms = np.expand_dims(validation_mixed_spectrograms, -1)  # append channels axis
		validation_speech_spectrograms = np.expand_dims(validation_speech_spectrograms, -1)  # append channels axis

		checkpoint = ModelCheckpoint(model_cache_path, verbose=1)

		lr_decay = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0, verbose=1)
		early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10, verbose=1)

		# tensorboard = TensorBoard(log_dir=tensorboard_dir, histogram_freq=0, write_graph=True, write_images=True)

		self.__model.fit(
			x=[train_mixed_spectrograms, train_video_samples],
			y=train_speech_spectrograms,

			validation_data=(
				[validation_mixed_spectrograms, validation_video_samples],
				validation_speech_spectrograms
			),

			batch_size=16, epochs=1000,
			# callbacks=[checkpoint, lr_decay, early_stopping, tensorboard],
			# verbose=1
			callbacks=[checkpoint, lr_decay, early_stopping],
			verbose=1
		)

	def predict(self, mixed_spectrograms, video_samples):
		mixed_spectrograms = np.expand_dims(mixed_spectrograms, -1)  # append channels axis
		speech_spectrograms = self.__model.predict([mixed_spectrograms, video_samples])

		return np.squeeze(speech_spectrograms)

	def evaluate(self, mixed_spectrograms, video_samples, speech_spectrograms):
		mixed_spectrograms = np.expand_dims(mixed_spectrograms, -1)  # append channels axis
		speech_spectrograms = np.expand_dims(speech_spectrograms, -1)  # append channels axis
		
		loss = self.__model.evaluate(x=[mixed_spectrograms, video_samples], y=speech_spectrograms)

		return loss

	@staticmethod
	def load(model_cache_path):
		model = load_model(model_cache_path)

		return SpeechEnhancementNetwork(model)

	def save(self, model_cache_path):
		self.__model.save(model_cache_path)
