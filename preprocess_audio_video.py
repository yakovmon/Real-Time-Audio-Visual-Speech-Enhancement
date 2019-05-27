import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import scipy.io.wavfile as wav
from data_processor import *
import threading
import time
import os

RATE = 16000
CHUNK_SIZE = 3200


def extract_frames(pathOut):

    print("working video")
    counter = 0
    video_cap = cv2.VideoCapture(0)
    success, image = video_cap.read()
    print('Read a new frame: ', success)
    r = cv2.selectROI(image, False)
    im_crop = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    height, width, channels = im_crop.shape
    slice_of_five_frames = np.zeros(shape=(int(height), int(width), 3, 5), dtype=np.float32)

    while counter < 5:

        video_cap.set(cv2.CAP_PROP_POS_MSEC, (counter*1))
        success, image = video_cap.read()
        print('Read a new frame: ', success)
        im_crop = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        slice_of_five_frames[:, :, :, counter] = im_crop

        # cv2.imwrite(pathOut + "\\frame%d.jpg" % counter, image) # save frame as JPEG file

        counter = counter + 1


    make_video(slice_of_five_frames)

    plot_slice(slice_of_five_frames)

    return slice_of_five_frames


def plot_slice(slice_of_five_frames):
    for i in range(2):
        im_gray = cv2.cvtColor(slice_of_five_frames[:, :, :, i], cv2.COLOR_BGR2GRAY)
        plt.imshow(im_gray, cmap='gray', vmin=0, vmax=255)
        plt.show()


def extract_audio():

    print("working audio")
    # initialize pyaudio
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)

    print("*start recording*")

    frames = []  # A python-list of chunks(numpy.ndarray)

    data = stream.read(CHUNK_SIZE)
    frames.append(np.fromstring(data, dtype=np.int16))

    print("*done recording*")

    # Convert the list of numpy-arrays into a 1D array (column-wise)
    numpy_data = np.hstack(frames)
    print(numpy_data.shape[0])

    # close stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    wav.write('out.wav', RATE, numpy_data)

    mixed_signal = AudioSignal.from_wav_file(numpy_data)
    mixed_spectrograms = preprocess_audio_signal(mixed_signal, 200, 1, 30)
    # return mixed_signal, mixed_spectrograms


def start_video():
    print("video thread")
    video_thread = threading.Thread(target=extract_frames("data"))
    video_thread.start()


def start_audio():
    video_thread = threading.Thread(target=extract_audio())
    video_thread.start()


def make_video(images, outimg=None, fps=5, size=None, is_color=True, format="XVID"):
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for i in range(5):
        image = cv2.cvtColor(images[:, :, :, i], cv2.COLOR_BGR2GRAY)
        if vid is None:
            if size is None:
                size = image.shape[1], image.shape[0]
            vid = VideoWriter("outvid.mp4", fourcc, float(fps), size, is_color)
        if size[0] != image.shape[1] and size[1] != image.shape[0]:
            img = resize(image, size)
        vid.write(image)
    vid.release()
    return vid


if __name__ == '__main__':

    global video_thread
    global audio_thread

    start_video()
    start_audio()

