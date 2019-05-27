import multiprocess
from collections import namedtuple

import numpy as np
import librosa

from facedetection.face_detection import FaceDetector
from mediaio.audio_io import AudioSignal, AudioMixer
from mediaio.video_io import VideoFileReader


def preprocess_video_sample(video_file_path, slice_duration_ms, mouth_height=128, mouth_width=128):
	print("preprocessing %s" % video_file_path)

	face_detector = FaceDetector()

	with VideoFileReader(video_file_path) as reader:
		frames = reader.read_all_frames(convert_to_gray_scale=True)

		mouth_cropped_frames = np.zeros(shape=(mouth_height, mouth_width, reader.get_frame_count()), dtype=np.float32)
		for i in range(reader.get_frame_count()):
			mouth_cropped_frames[:, :, i] = face_detector.crop_mouth(frames[i], bounding_box_shape=(mouth_width, mouth_height))

		frames_per_slice = int((float(slice_duration_ms) / 1000) * reader.get_frame_rate())
		n_slices = int(float(reader.get_frame_count()) / frames_per_slice)

		slices = [
			mouth_cropped_frames[:, :, (i * frames_per_slice):((i + 1) * frames_per_slice)]
			for i in range(n_slices)
		]

		return np.stack(slices), reader.get_frame_rate()


def preprocess_audio_signal(audio_signal, slice_duration_ms, n_video_slices, video_frame_rate):
	samples_per_slice = int((float(slice_duration_ms) / 1000) * audio_signal.get_sample_rate())
	signal_length = samples_per_slice * n_video_slices

	if audio_signal.get_number_of_samples() < signal_length:
		audio_signal.pad_with_zeros(signal_length)
	else:
		audio_signal.truncate(signal_length)

	n_fft = int(float(audio_signal.get_sample_rate()) / video_frame_rate)
	hop_length = int(n_fft / 4)

	mel_spectrogram, phase = signal_to_spectrogram(audio_signal, n_fft, hop_length, mel=True, db=True)

	spectrogram_samples_per_slice = int(samples_per_slice / hop_length)
	n_slices = int(mel_spectrogram.shape[1] / spectrogram_samples_per_slice)

	slices = [
		mel_spectrogram[:, (i * spectrogram_samples_per_slice):((i + 1) * spectrogram_samples_per_slice)]
		for i in range(n_slices)
	]

	return np.stack(slices)


def reconstruct_speech_signal(mixed_signal, speech_spectrograms, video_frame_rate):
	n_fft = int(float(mixed_signal.get_sample_rate()) / video_frame_rate)
	hop_length = int(n_fft / 4)

	_, original_phase = signal_to_spectrogram(mixed_signal, n_fft, hop_length, mel=True, db=True)

	speech_spectrogram = np.concatenate(list(speech_spectrograms), axis=1)

	spectrogram_length = min(speech_spectrogram.shape[1], original_phase.shape[1])
	speech_spectrogram = speech_spectrogram[:, :spectrogram_length]
	original_phase = original_phase[:, :spectrogram_length]

	return reconstruct_signal_from_spectrogram(
		speech_spectrogram, original_phase, mixed_signal.get_sample_rate(), n_fft, hop_length, mel=True, db=True
	)


def signal_to_spectrogram(audio_signal, n_fft, hop_length, mel=True, db=True):
	signal = audio_signal.get_data(channel_index=0)
	D = librosa.core.stft(signal.astype(np.float64), n_fft=n_fft, hop_length=hop_length)
	magnitude, phase = librosa.core.magphase(D)

	if mel:
		mel_filterbank = librosa.filters.mel(
			sr=audio_signal.get_sample_rate(),
			n_fft=n_fft,
			n_mels=80,
			fmin=0,
			fmax=8000
		)

		magnitude = np.dot(mel_filterbank, magnitude)

	if db:
		magnitude = librosa.amplitude_to_db(magnitude)

	return magnitude, phase


def reconstruct_signal_from_spectrogram(magnitude, phase, sample_rate, n_fft, hop_length, mel=True, db=True):
	if db:
		magnitude = librosa.db_to_amplitude(magnitude)

	if mel:
		mel_filterbank = librosa.filters.mel(
			sr=sample_rate,
			n_fft=n_fft,
			n_mels=80,
			fmin=0,
			fmax=8000
		)

		magnitude = np.dot(np.linalg.pinv(mel_filterbank), magnitude)

	signal = librosa.istft(magnitude * phase, hop_length=hop_length)

	return AudioSignal(signal, sample_rate)


def preprocess_audio_pair(speech_file_path, noise_file_path, slice_duration_ms, n_video_slices, video_frame_rate):
	print("preprocessing pair: %s, %s" % (speech_file_path, noise_file_path))

	speech_signal = AudioSignal.from_wav_file(speech_file_path)
	noise_signal = AudioSignal.from_wav_file(noise_file_path)

	while noise_signal.get_number_of_samples() < speech_signal.get_number_of_samples():
		noise_signal = AudioSignal.concat([noise_signal, noise_signal])

	noise_signal.truncate(speech_signal.get_number_of_samples())

	factor = AudioMixer.snr_factor(speech_signal, noise_signal, snr_db=0)
	noise_signal.amplify_by_factor(factor)

	mixed_signal = AudioMixer.mix([speech_signal, noise_signal], mixing_weights=[1, 1])

	mixed_spectrograms = preprocess_audio_signal(mixed_signal, slice_duration_ms, n_video_slices, video_frame_rate)
	speech_spectrograms = preprocess_audio_signal(speech_signal, slice_duration_ms, n_video_slices, video_frame_rate)
	noise_spectrograms = preprocess_audio_signal(noise_signal, slice_duration_ms, n_video_slices, video_frame_rate)

	return mixed_spectrograms, speech_spectrograms, noise_spectrograms, mixed_signal


Sample = namedtuple('Sample', [
	'speaker_id',
	'video_file_path',
	'speech_file_path',
	'noise_file_path',
	'video_samples',
	'mixed_spectrograms',
	'speech_spectrograms',
	'noise_spectrograms',
	'mixed_signal',
	'video_frame_rate'
])


def preprocess_sample(speech_entry, noise_file_path, slice_duration_ms=200):
	print("preprocessing sample: %s, %s, %s..." % (speech_entry.video_path, speech_entry.audio_path, noise_file_path))

	video_samples, video_frame_rate = preprocess_video_sample(speech_entry.video_path, slice_duration_ms)
	mixed_spectrograms, speech_spectrograms, noise_spectrograms, mixed_signal = preprocess_audio_pair(
		speech_entry.audio_path, noise_file_path, slice_duration_ms, video_samples.shape[0], video_frame_rate
	)

	n_slices = min(video_samples.shape[0], mixed_spectrograms.shape[0])

	return Sample(
		speaker_id=speech_entry.speaker_id,
		video_file_path=speech_entry.video_path,
		speech_file_path=speech_entry.audio_path,
		noise_file_path=noise_file_path,
		video_samples=video_samples[:n_slices],
		mixed_spectrograms=mixed_spectrograms[:n_slices],
		speech_spectrograms=speech_spectrograms[:n_slices],
		noise_spectrograms=noise_spectrograms[:n_slices],
		mixed_signal=mixed_signal,
		video_frame_rate=video_frame_rate
	)


def try_preprocess_sample(sample_paths):
	try:
		return preprocess_sample(*sample_paths)

	except Exception as e:
		print("failed to preprocess %s (%s)" % (sample_paths, e))
		return None


def preprocess_data(speech_entries, noise_file_paths):
	print("preprocessing data...")

	sample_paths = zip(speech_entries, noise_file_paths)

	thread_pool = multiprocess.Pool(16)
	samples = thread_pool.map(try_preprocess_sample, sample_paths)
	samples = [p for p in samples if p is not None]

	return samples


class VideoNormalizer(object):

	def __init__(self, video_samples):
		# video_samples: slices x height x width x frames_per_slice
		self.__mean_image = np.mean(video_samples, axis=(0, 3))
		self.__std_image = np.std(video_samples, axis=(0, 3))

	def normalize(self, video_samples):
		for s in range(video_samples.shape[0]):
			for f in range(video_samples.shape[3]):
				video_samples[s, :, :, f] -= self.__mean_image
				video_samples[s, :, :, f] /= self.__std_image
