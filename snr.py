import argparse
import os

from mediaio.audio_io import AudioSignal

import numpy as np


def evaluate(enhancement_dir):
	noisy_snr_dbs = []
	snr_dbs = []

	speaker_ids = os.listdir(enhancement_dir)
	for speaker_id in speaker_ids:
		for sample_dir_name in os.listdir(os.path.join(enhancement_dir, speaker_id)):
			print('evaluating snr of %s' % sample_dir_name)

			source_path = os.path.join(enhancement_dir, speaker_id, sample_dir_name, 'source.wav')
			mixture_path = os.path.join(enhancement_dir, speaker_id, sample_dir_name, 'mixture.wav')
			enhanced_path = os.path.join(enhancement_dir, speaker_id, sample_dir_name, 'enhanced.wav')

			source_signal = AudioSignal.from_wav_file(source_path)
			mixture_signal = AudioSignal.from_wav_file(mixture_path)
			enhanced_signal = AudioSignal.from_wav_file(enhanced_path)

			truncate_longer_signal(mixture_signal, source_signal)

			s = source_signal.get_data()
			n = mixture_signal.get_data() - source_signal.get_data()

			noisy_snr = np.var(s) / np.var(n)
			noisy_snr_db = 10 * np.log10(noisy_snr)
			print('noisy snr db: %f' % noisy_snr_db)

			noisy_snr_dbs.append(noisy_snr_db)

			truncate_longer_signal(enhanced_signal, source_signal)

			s = source_signal.get_data()
			e = enhanced_signal.get_data()
			residual_noise = e - s

			snr = np.var(s) / np.var(residual_noise)
			snr_db = 10 * np.log10(snr)
			print('snr db: %f' % snr_db)

			snr_dbs.append(snr_db)

	print('mean noisy snr db: %f' % np.mean(noisy_snr_dbs))
	print('mean snr db: %f' % np.mean(snr_dbs))


def truncate_longer_signal(signal1, signal2):
	if signal1.get_number_of_samples() < signal2.get_number_of_samples():
		signal2.truncate(signal1.get_number_of_samples())

	else:
		signal1.truncate(signal2.get_number_of_samples())


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("enhancement_dir", type=str)
	args = parser.parse_args()

	evaluate(args.enhancement_dir)


if __name__ == "__main__":
	main()
