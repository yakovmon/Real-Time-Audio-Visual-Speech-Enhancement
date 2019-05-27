from os import listdir, system
from os.path import isfile, join
import re


def extract_audio(input_video_file_path):
    video_parts = [f for f in listdir(input_video_file_path) if isfile(join(input_video_file_path, f))]
    for f in video_parts:
        if f.endswith(".mp4"):
            index = f.index('.')
            name = f[:index]
            system("ffmpeg -i /cs/engproj/322/raw_video/new_test_video/raw_video_test/video/{0} -vn -ac 1 -ar 16000"
                   " /cs/engproj/322/raw_video/new_test_video/raw_video_test/audio/{1}.wav".format(f, name))
        else:
            continue


def main():
        extract_audio("/cs/engproj/322/raw_video/new_test_video/raw_video_test/video")


if __name__ == '__main__':
    main()
