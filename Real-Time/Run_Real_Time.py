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

from Video_Preprocess import *
from Audio_Preprocess import *


def predict(args):
    assets = AssetManager(args.base_dir)
    storage = PredictionStorage(assets.create_prediction_storage(args.model, args.data_name))

    # Change path of args.model manually
    network = SpeechEnhancementNetwork.load(assets.get_model_cache_path(args.model))

    with open(assets.get_normalization_cache_path(args.model), 'rb') as normalization_fd:
        video_normalizer = pickle.load(normalization_fd)

    try:

        video_normalizer.normalize(extract_frames("data"))

        predicted_speech_spectrograms = network.predict(extract_audio()[1], extract_frames("data"))

        predicted_speech_signal = data_processor.reconstruct_speech_signal(
            extract_audio()[0], predicted_speech_spectrograms, 30)

        predicted_speech_signal.save_to_wav_file("enhanced.wav")

    except Exception:
        logging.exception("failed to predict %s. skipping" % "test")


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
        enhanced_speech_video_path = os.path.join(sample_prediction_dir,
                                                  "enhanced" + video_extension)

        ffmpeg.merge(sample.video_file_path, mixture_audio_path, mixture_video_path)
        ffmpeg.merge(sample.video_file_path, enhanced_speech_audio_path, enhanced_speech_video_path)


def main():
    predict("args")
