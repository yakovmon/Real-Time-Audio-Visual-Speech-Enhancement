import os
import glob
import random
from collections import namedtuple


AudioVisualEntry = namedtuple('AudioVisualEntry', ['speaker_id', 'audio_path', 'video_path'])


class AudioVisualDataset:

	def __init__(self, base_path):
		self._base_path = base_path

	def subset(self, speaker_ids, max_files=None, shuffle=False):
		entries = []

		for speaker_id in speaker_ids:
			audio_paths = glob.glob(os.path.join(self._base_path, speaker_id, 'audio', '*.wav'))

			for audio_path in audio_paths:
				entry = AudioVisualEntry(speaker_id, audio_path, AudioVisualDataset.__audio_to_video_path(audio_path))
				entries.append(entry)

		if shuffle:
			random.shuffle(entries)

		return entries[:max_files]

	def list_speakers(self):
		return os.listdir(self._base_path)

	@staticmethod
	def __audio_to_video_path(audio_path):
		path_names = audio_path.split('/')
		if path_names[-2] != 'audio':
			raise Exception('invalid audio-video path conversion')

		path_names[-2] = 'video'

		return glob.glob(os.path.splitext('/'.join(path_names))[0] + ".*")[0]


class AudioDataset:

	def __init__(self, base_paths):
		self._base_paths = base_paths

	def subset(self, max_files=None, shuffle=False):
		audio_file_paths = [os.path.join(d, f) for d in self._base_paths for f in os.listdir(d)]

		if shuffle:
			random.shuffle(audio_file_paths)

		return audio_file_paths[:max_files]
