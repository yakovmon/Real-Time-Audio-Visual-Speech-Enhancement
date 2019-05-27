import subprocess


def downsample(input_audio_file_path, output_audio_file_path, sample_rate):
	subprocess.check_call(
		["ffmpeg", "-i", input_audio_file_path, "-ar", str(sample_rate), output_audio_file_path, "-y"]
	)


def merge(input_video_file_path, input_audio_file_path, output_video_file_path):
	subprocess.check_call([
		"ffmpeg", '-hide_banner', '-loglevel', 'panic', "-i", input_video_file_path, "-i", input_audio_file_path,
		 "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0", output_video_file_path
	])
