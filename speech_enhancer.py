import argparse
import os
import logging
import pickle
import random
from datetime import datetime

import numpy as np

import data_processor
from dataset import AudioVisualDataset, AudioDataset
from network import SpeechEnhancementNetwork
from shutil import copy2
from mediaio import ffmpeg

def preprocess(args):
	assets = AssetManager(args.base_dir)
	speaker_ids = list_speakers(args)

	speech_entries, noise_file_paths = list_data(
		args.dataset_dir, speaker_ids, args.noise_dirs, max_files=1000, shuffle=True, augmentation_factor=1
	)

	samples = data_processor.preprocess_data(speech_entries, noise_file_paths)

	with open(assets.get_preprocessed_blob_path(args.data_name), 'wb') as preprocessed_fd:
		pickle.dump(samples, preprocessed_fd)


def train(args):
	assets = AssetManager(args.base_dir)
	assets.create_model(args.model)

	train_preprocessed_blob_paths = [assets.get_preprocessed_blob_path(d) for d in args.train_data_names]
	validation_preprocessed_blob_paths = [assets.get_preprocessed_blob_path(d) for d in args.validation_data_names]

	train_samples = load_preprocessed_blobs(train_preprocessed_blob_paths)
	train_video_samples, train_mixed_spectrograms, train_speech_spectrograms = make_sample_set(train_samples)

	validation_samples = load_preprocessed_blobs(validation_preprocessed_blob_paths)
	validation_video_samples, validation_mixed_spectrograms, validation_speech_spectrograms = make_sample_set(validation_samples)

	video_normalizer = data_processor.VideoNormalizer(train_video_samples)
	video_normalizer.normalize(train_video_samples)
	video_normalizer.normalize(validation_video_samples)

	with open(assets.get_normalization_cache_path(args.model), 'wb') as normalization_fd:
		pickle.dump(video_normalizer, normalization_fd)

	network = SpeechEnhancementNetwork.build(train_mixed_spectrograms.shape[1:], train_video_samples.shape[1:])
	network.train(
		train_mixed_spectrograms, train_video_samples, train_speech_spectrograms,
		validation_mixed_spectrograms, validation_video_samples, validation_speech_spectrograms,
		assets.get_model_cache_path(args.model), assets.get_tensorboard_dir(args.model)
	)

	network.save(assets.get_model_cache_path(args.model))


def predict(args):
	assets = AssetManager(args.base_dir)
	storage = PredictionStorage(assets.create_prediction_storage(args.model, args.data_name))
	network = SpeechEnhancementNetwork.load(assets.get_model_cache_path(args.model))

	with open(assets.get_normalization_cache_path(args.model), 'rb') as normalization_fd:
		video_normalizer = pickle.load(normalization_fd)

	samples = load_preprocessed_blob(assets.get_preprocessed_blob_path(args.data_name))
	for sample in samples:
		try:
			print("predicting (%s, %s)..." % (sample.video_file_path, sample.noise_file_path))

			video_normalizer.normalize(sample.video_samples)

			loss = network.evaluate(sample.mixed_spectrograms, sample.video_samples, sample.speech_spectrograms)
			print("loss: %f" % loss)

			predicted_speech_spectrograms = network.predict(sample.mixed_spectrograms, sample.video_samples)

			predicted_speech_signal = data_processor.reconstruct_speech_signal(
				sample.mixed_signal, predicted_speech_spectrograms, sample.video_frame_rate
			)

			storage.save_prediction(sample, predicted_speech_signal)

		except Exception:
			logging.exception("failed to predict %s. skipping" % sample.video_file_path)


class AssetManager:

	def __init__(self, base_dir):
		self.__base_dir = base_dir

		self.__cache_dir = os.path.join(self.__base_dir, 'cache')
		if not os.path.exists(self.__cache_dir):
			os.mkdir(self.__cache_dir)

		self.__preprocessed_dir = os.path.join(self.__cache_dir, 'preprocessed')
		if not os.path.exists(self.__preprocessed_dir):
			os.mkdir(self.__preprocessed_dir)

		self.__models_dir = os.path.join(self.__cache_dir, 'models')
		if not os.path.exists(self.__models_dir):
			os.mkdir(self.__models_dir)

		self.__out_dir = os.path.join(self.__base_dir, 'out')
		if not os.path.exists(self.__out_dir):
			os.mkdir(self.__out_dir)

	def get_preprocessed_blob_path(self, data_name):
		return os.path.join(self.__preprocessed_dir, data_name + '.pkl')

	def create_model(self, model_name):
		model_dir = os.path.join(self.__models_dir, model_name)
		if not os.path.exists(model_dir):
			os.mkdir(model_dir)

	def get_model_cache_path(self, model_name):
		model_dir = os.path.join(self.__models_dir, model_name)
		return os.path.join(model_dir, 'model.h5py')

	def get_normalization_cache_path(self, model_name):
		model_dir = os.path.join(self.__models_dir, model_name)
		return os.path.join(model_dir, 'normalization.pkl')

	def get_tensorboard_dir(self, model_name):
		model_dir = os.path.join(self.__models_dir, model_name)
		tensorboard_dir = os.path.join(model_dir, 'tensorboard')

		if not os.path.exists(tensorboard_dir):
			os.mkdir(tensorboard_dir)

		return tensorboard_dir

	def create_prediction_storage(self, model_name, data_name):
		prediction_dir = os.path.join(self.__out_dir, model_name, data_name)
		if not os.path.exists(prediction_dir):
			os.makedirs(prediction_dir)

		return prediction_dir


class PredictionStorage(object):

	def __init__(self, storage_dir):
		self.__base_dir = os.path.join(storage_dir, '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.now()))
		os.mkdir(self.__base_dir)

	def __create_speaker_dir(self, speaker_id):
		speaker_dir = os.path.join(self.__base_dir, speaker_id)

		if not os.path.exists(speaker_dir):
			os.mkdir(speaker_dir)

		return speaker_dir

	def save_prediction(self, sample, predicted_speech_signal):
		speaker_dir = self.__create_speaker_dir(sample.speaker_id)

		speech_name = os.path.splitext(os.path.basename(sample.video_file_path))[0]
		noise_name = os.path.splitext(os.path.basename(sample.noise_file_path))[0]

		sample_prediction_dir = os.path.join(speaker_dir, speech_name + "_" + noise_name)
		os.mkdir(sample_prediction_dir)

		mixture_audio_path = os.path.join(sample_prediction_dir, "mixture.wav")
		enhanced_speech_audio_path = os.path.join(sample_prediction_dir, "enhanced.wav")
		source_audio_path = os.path.join(sample_prediction_dir, "source.wav")
		noise_audio_path = os.path.join(sample_prediction_dir, "noise.wav")

		copy2(sample.speech_file_path, source_audio_path)
		copy2(sample.noise_file_path, noise_audio_path)

		sample.mixed_signal.save_to_wav_file(mixture_audio_path)
		predicted_speech_signal.save_to_wav_file(enhanced_speech_audio_path)

		video_extension = os.path.splitext(os.path.basename(sample.video_file_path))[1]
		mixture_video_path = os.path.join(sample_prediction_dir, "mixture" + video_extension)
		enhanced_speech_video_path = os.path.join(sample_prediction_dir, "enhanced" + video_extension)

		ffmpeg.merge(sample.video_file_path, mixture_audio_path, mixture_video_path)
		ffmpeg.merge(sample.video_file_path, enhanced_speech_audio_path, enhanced_speech_video_path)


def list_speakers(args):
	if args.speakers is None:
		dataset = AudioVisualDataset(args.dataset_dir)
		speaker_ids = dataset.list_speakers()
	else:
		speaker_ids = args.speakers

	if args.ignored_speakers is not None:
		for speaker_id in args.ignored_speakers:
			speaker_ids.remove(speaker_id)

	return speaker_ids


def list_data(dataset_dir, speaker_ids, noise_dirs, max_files=None, shuffle=True, augmentation_factor=1):
	speech_dataset = AudioVisualDataset(dataset_dir)
	speech_subset = speech_dataset.subset(speaker_ids, max_files, shuffle)

	noise_dataset = AudioDataset(noise_dirs)
	noise_file_paths = noise_dataset.subset(max_files, shuffle)

	n_files = min(len(speech_subset), len(noise_file_paths))

	speech_entries = speech_subset[:n_files]
	noise_file_paths = noise_file_paths[:n_files]

	all_speech_entries = speech_entries
	all_noise_file_paths = noise_file_paths

	for i in range(augmentation_factor - 1):
		all_speech_entries += speech_entries
		all_noise_file_paths += random.sample(noise_file_paths, len(noise_file_paths))

	return all_speech_entries, all_noise_file_paths


def load_preprocessed_blob(preprocessed_blob_path):
	print("loading preprocessed samples from %s" % preprocessed_blob_path)

	with open(preprocessed_blob_path, 'rb') as preprocessed_fd:
		samples = pickle.load(preprocessed_fd)

	return samples


def load_preprocessed_blobs(preprocessed_blob_paths, max_samples_per_blob=None):
	all_samples = []

	for preprocessed_blob_path in preprocessed_blob_paths:
		all_samples += load_preprocessed_blob(preprocessed_blob_path)[:max_samples_per_blob]

	return all_samples


def make_sample_set(samples, max_samples=None):
	if max_samples is not None:
		n_samples = min(len(samples), max_samples)
	else:
		n_samples = len(samples)

	samples = random.sample(samples, n_samples)

	video_samples = np.concatenate([sample.video_samples for sample in samples], axis=0)
	mixed_spectrograms = np.concatenate([sample.mixed_spectrograms for sample in samples], axis=0)
	speech_spectrograms = np.concatenate([sample.speech_spectrograms for sample in samples], axis=0)

	permutation = np.random.permutation(video_samples.shape[0])
	video_samples = video_samples[permutation]
	mixed_spectrograms = mixed_spectrograms[permutation]
	speech_spectrograms = speech_spectrograms[permutation]

	return (
		video_samples,
		mixed_spectrograms,
		speech_spectrograms
	)


def main():
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument('-bd', '--base_dir', type=str, required=True)

	action_parsers = parser.add_subparsers()

	preprocess_parser = action_parsers.add_parser("preprocess")
	preprocess_parser.add_argument('-dn', '--data_name', type=str, required=True)
	preprocess_parser.add_argument('-ds', '--dataset_dir', type=str, required=True)
	preprocess_parser.add_argument('-n', '--noise_dirs', nargs='+', type=str, required=True)
	preprocess_parser.add_argument('-s', '--speakers', nargs='+', type=str)
	preprocess_parser.add_argument('-is', '--ignored_speakers', nargs='+', type=str)
	preprocess_parser.set_defaults(func=preprocess)

	train_parser = action_parsers.add_parser("train")
	train_parser.add_argument('-mn', '--model', type=str, required=True)
	train_parser.add_argument('-tdn', '--train_data_names', nargs='+', type=str, required=True)
	train_parser.add_argument('-vdn', '--validation_data_names', nargs='+', type=str, required=True)
	train_parser.add_argument('-g', '--gpus', type=int, default=1)
	train_parser.set_defaults(func=train)

	predict_parser = action_parsers.add_parser("predict")
	predict_parser.add_argument('-mn', '--model', type=str, required=True)
	predict_parser.add_argument('-dn', '--data_name', type=str, required=True)
	predict_parser.add_argument('-g', '--gpus', type=int, default=1)
	predict_parser.set_defaults(func=predict)

	args = parser.parse_args()
	args.func(args)


if __name__ == "__main__":
	main()
