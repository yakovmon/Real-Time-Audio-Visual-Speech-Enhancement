from os import listdir, system
from os.path import isfile, join
import re


def extract_audio(input_video_file_path):
    video_parts = [f for f in listdir(input_video_file_path) if isfile(join(input_video_file_path, f))]
    for f in video_parts:
        if f.endswith(".mp4"):
            numbers = re.findall('\d+', f)
            system("ffmpeg -i /cs/engproj/322/raw_video/video_parts/{0} -vn -ac 1 -ar 16000"
                   " /cs/engproj/322/raw_video/audio_parts/part{1}.wav".format(f, numbers[0]))
        else:
            continue


def main():
        extract_audio("/cs/engproj/322/raw_video/video_parts")


if __name__ == '__main__':
    main()
