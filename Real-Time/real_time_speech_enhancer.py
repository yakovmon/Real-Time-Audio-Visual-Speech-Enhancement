import cv2
import numpy as np
from facedetection.face_detection import FaceDetector
from multiprocessing import Process, Queue, Lock
import scipy.io.wavfile as wav
import subprocess
from data_processor import preprocess_audio_signal, reconstruct_speech_signal
from mediaio.audio_io import AudioSignal
from network import SpeechEnhancementNetwork
import pickle
import logging
import wave
import os
import argparse
from datetime import datetime
import sys
import time
import math
import sounddevice as sd
import pyaudio
import matplotlib.pyplot as plt


class VideoProcess:
    """
    VideoProcess class based on openCV.
    """

    # NUMBER_OF_FRAMES = 6
    NUMBER_OF_FRAMES = 12
    # NUMBER_OF_FRAMES = 30
    # NUMBER_OF_FRAMES = 150

    def __init__(self, video_path):

        print("*Initialize VideoProcess class*\n")

        self.open = True
        self.frames_counter = 0
        self.slice_counter = 0
        self.frame_list = []
        self.video_path = video_path

    def capture_frames(self, queue, lock):
        video_cap = cv2.VideoCapture(self.video_path)
        frames_count = float(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slice_counter = math.ceil(frames_count / VideoProcess.NUMBER_OF_FRAMES)

        with lock:
            print("First frame time: " + str(datetime.now()))
            print("*Start capture video frames*")
            print("*******************************Video Parameters*******************************")
            print("Video - Number of slices: " + str(self.slice_counter))
            print("FPS " + str(video_cap.get(cv2.CAP_PROP_FPS)))
            print("HEIGHT " + str(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            print("WIDTH " + str(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            print("FRAME_COUNT " + str(frames_count))
            print("******************************************************************************\n")

        while self.open:
            if self.frames_counter == 0:
                start_time = time.time()
            success, video_frame = video_cap.read()
            time.sleep(0.029)
            if self.frames_counter == 5:
                with lock:
                    print("--- FPS Check: %s seconds ---" % (time.time() - start_time))
            if success:
                self.frames_counter += 1
                self.frame_list.append(video_frame)
                if self.frames_counter % VideoProcess.NUMBER_OF_FRAMES == 0:
                    self.slice_counter -= 1
                    queue.put((self.frame_list, self.slice_counter))
                    self.frame_list = []
            else:
                break

        if self.frames_counter % VideoProcess.NUMBER_OF_FRAMES != 0:
            with lock:
                print("Total number of frames: " + str(self.frames_counter))
            self.slice_counter -= 1
            queue.put((self.frame_list, self.slice_counter))

        with lock:
            print("*Exit video stream*\n")

        video_cap.release()
        cv2.destroyAllWindows()


class AudioProcess:
    """
    VideoProcess class based on wav library.
    """

    RATE = 16000

    # CHUNK_SIZE = 3200
    CHUNK_SIZE = 6400
    # CHUNK_SIZE = 16000
    # CHUNK_SIZE = 80000

    def __init__(self, audio_path):

        print("*Initialize AudioProcess class*\n")

        self.open = True
        self.rate = AudioProcess.RATE
        self.frames_per_buffer = AudioProcess.CHUNK_SIZE
        self.audio_path = audio_path
        self.slice_counter = 0

    def capture_frames(self, queue, lock):

        wf = wave.open(self.audio_path, 'rb')
        self.slice_counter = math.ceil(float(wf.getnframes()) / self.frames_per_buffer)

        with lock:
            print("*Start capture audio frames*")
            print("*******************************Audio Parameters*******************************")
            print("Audio - Number of slices: " + str(self.slice_counter))
            print("wf.getparams " + str(wf.getparams()))
            print("******************************************************************************\n")

        while self.open:
            raw_data = wf.readframes(self.frames_per_buffer)
            if raw_data == b'':  # The file is over
                break
            self.slice_counter -= 1
            queue.put((raw_data, self.slice_counter))

        with lock:
            print("*Exit audio stream*\n")

        wf.close()


class RunPredict:
    """
    This class runs real time predict.
    """

    FRAMES_PER_SLICE = 6

    # NUMBER_OF_FRAMES = 6
    NUMBER_OF_FRAMES = 12
    # NUMBER_OF_FRAMES = 30
    # NUMBER_OF_FRAMES = 150

    NUMBER_OF_SLICES = int(NUMBER_OF_FRAMES / FRAMES_PER_SLICE)

    def __init__(self, network, video_path, storage_dir, width=128, height=128):

        print("*Initialize RunPredict class*\n")

        self.face_detector = FaceDetector()
        self.path_video_writer_path = os.path.join(storage_dir, "video_input_realtime.avi")
        self.video_path = video_path
        self.frame_size = (width, height)
        self.video_writer_frame_size = (1280, 720)
        self.fps, self.bounding_box = self.detect_bounding_box()
        self.open = True
        self.slice_of_frames = \
            np.zeros(shape=(height, width, RunPredict.NUMBER_OF_FRAMES), dtype=np.float32)
        self.frames_counter = 0
        self.slice_duration_ms = 200
        self.n_video_slices = RunPredict.NUMBER_OF_SLICES
        self.network = network
        self.save_predicted = []
        self.save_original = []
        self.save_signal = []
        self.num_iteration = 0
        self.sample_type_info = np.iinfo(np.int16)

    def detect_bounding_box(self):

        print("*Detect bounding box*")

        video_cap = cv2.VideoCapture(self.video_path)
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        success, image = video_cap.read()

        try:
            if success:
                bounding_box = self.face_detector.detect_mouth(image,
                                                               bounding_box_shape=self.frame_size)
                print("*Finish to create bounding box*\n")

            else:
                raise Exception('*Error - Detecting bounding box*')

        except Exception as error:
            print('Caught this error: ' + repr(error))
            raise

        finally:
            video_cap.release()
            cv2.destroyAllWindows()

        return fps, bounding_box

    def run_pre_process(self, v_queue, a_queue, predict_queue, video_normalizer, lock):

        with lock:
            print("*Start pre-process*\n")

        video_out = cv2.VideoWriter(self.path_video_writer_path,
                                    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                    self.fps, self.frame_size)

        while self.open:

            video_frames_list, video_slice_number = v_queue.get()
            audio_frames_list, audio_slice_number = a_queue.get()

            # Video pre-process
            for frame in video_frames_list:
                im_crop = self.face_detector.crop_mouth(frame, self.bounding_box)
                video_out.write(im_crop)

                im_gray = cv2.cvtColor(im_crop, cv2.COLOR_BGR2GRAY)
                self.slice_of_frames[:, :, self.frames_counter] = im_gray
                self.frames_counter += 1

            slices = [
                self.slice_of_frames
                [:, :, (i * RunPredict.FRAMES_PER_SLICE):((i + 1) * RunPredict.FRAMES_PER_SLICE)]
                for i in range(RunPredict.NUMBER_OF_SLICES)
            ]

            slices = np.stack(slices)
            video_normalizer.normalize(slices)

            # Audio pre-process
            data = np.fromstring(audio_frames_list, dtype=np.int16)
            mixed_signal = AudioSignal(data, 16000)

            self.num_iteration += 1

            mixed_spectrograms = preprocess_audio_signal(mixed_signal, self.slice_duration_ms,
                                                         self.n_video_slices, self.fps)

            # Predict
            predict_queue.put((slices, mixed_signal, mixed_spectrograms, int(video_slice_number),
                               int(audio_slice_number)))

            self.slice_of_frames = \
                np.zeros(shape=(128, 128, RunPredict.NUMBER_OF_FRAMES), dtype=np.float32)
            self.frames_counter = 0

            if (v_queue.empty() and a_queue.empty()) or audio_slice_number == 0\
                    or video_slice_number == 0:

                with lock:
                    print("****************************************************************")
                    print("Video - slice number: " + str(video_slice_number))
                    print("Audio - slice number: " + str(audio_slice_number))
                    print("Predict - number of iterations: " + str(self.num_iteration))
                    print("****************************************************************")

                if not v_queue.empty():
                    video_slice_number -= 1
                    v_queue.get()

                elif not a_queue.empty():
                    audio_slice_number -= 1
                    a_queue.get()

                v_queue.close()
                a_queue.close()

                with lock:
                    print("*Video queue and Audio queue are empty*\n")
                break

    def predict(self, predict_queue, play_queue, lock):

        with lock:
            print("*Start Predict*\n")

        counter = 0
        while True:

            video_data, mixed_signal, mixed_spectrograms, video_slice_number, audio_slice_number = predict_queue.get()

            # Spectrogram Test
            # self.collect_frames_for_saving(mixed_signal, self.save_original, object_flag=False)
            # after_spectrograms = reconstruct_speech_signal(mixed_signal, mixed_spectrograms, self.fps)
            # self.collect_frames_for_saving(after_spectrograms, self.save_signal, object_flag=True)

            counter += 1
            try:

                predicted_speech_spectrograms = self.network.predict(mixed_spectrograms, video_data)
                predicted_speech_signal = reconstruct_speech_signal(mixed_signal, predicted_speech_spectrograms,
                                                                    self.fps)

                self.collect_frames_for_saving(predicted_speech_signal, self.save_predicted,
                                               object_flag=True)

                play_queue.put((predicted_speech_signal, video_slice_number, audio_slice_number))

                if (video_slice_number == 0 and predict_queue.empty()) or\
                        (audio_slice_number == 0 and predict_queue.empty()):
                    with lock:
                        print("*Predict queue is empty*\n")
                        print("Predict: " + str(counter))
                    predict_queue.close()
                    break

            except Exception:
                    logging.exception("Failed to predict")

    def play(self, play_queue, lock):

        # Live audio stream
        counter = 0
        with lock:
            print("*Start Play*\n")

        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, output=True)

        # sd._initialize()

        while True:
            counter += 1

            predicted_speech_signal, video_slice_number, audio_slice_number = play_queue.get()
            clean_audio = predicted_speech_signal.get_data() \
                .clip(self.sample_type_info.min, self.sample_type_info.max).astype(np.int16)

            if counter == 1:
                with lock:
                    print("First play time: " + str(datetime.now()))

            stream.write(clean_audio.tobytes(), exception_on_underflow=False)

            # sd.play(clean_audio, 16000, blocking=True)

            if (video_slice_number == 0 and play_queue.empty()) or (audio_slice_number == 0 and play_queue.empty()):
                with lock:
                    print("*play queue is empty*\n")
                    print("Play: " + str(counter))
                play_queue.close()

                stream.stop_stream()
                stream.close()
                p.terminate()

                break

    def collect_frames_for_saving(self, data, list_name, object_flag=False):
        if object_flag:
            data = data.get_data().clip(self.sample_type_info.min,
                                        self.sample_type_info.max).astype(np.int16)
            list_name.append(data)
        else:
            list_name.append(data.get_data())

    # TODO: Move to PredictionStorage class
    def save_files(self, storage):

        rate = 16000

        print("*Saving wav files*")

        prediction_dir = storage.storage_dir

        # audio_input_real_time = os.path.join(prediction_dir, "audio_input_real_time.wav")
        video_input_realtime = os.path.join(prediction_dir, "video_input_realtime.avi")
        # audio_reconstruct_real_time = os.path.join(prediction_dir, "audio_reconstruct_real_time.wav")
        enhanced_real_time = os.path.join(prediction_dir, "enhanced_real_time.wav")
        enhanced_video_output = os.path.join(prediction_dir, "enhanced_video_output.avi")

        # wav.write(audio_input_real_time, rate, np.hstack(self.save_original))
        # wav.write(audio_reconstruct_real_time, rate, np.hstack(self.save_signal))
        wav.write(enhanced_real_time, rate, np.hstack(self.save_predicted))

        print("*Mixing*")

        cmd = "ffmpeg -i {} -i {} -codec copy {}".format(video_input_realtime,
                                                         enhanced_real_time, enhanced_video_output)

        subprocess.call(cmd, shell=True)


class AssetManager:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def get_model_cache_path(self, model_name):
        return os.path.join(model_name, 'model.h5py')

    def get_normalization_cache_path(self, model_name):
        return os.path.join(model_name, 'normalization.pkl')

    def get_video_cache_path(self, video_audio_dir):
        return os.path.join(video_audio_dir, 'high_volume.mp4')

    def get_audio_cache_path(self, video_audio_dir):
        return os.path.join(video_audio_dir, 'high_volume.wav')


class PredictionStorage:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.storage_dir = os.path.join(self.base_dir, '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.now()))
        os.mkdir(self.storage_dir)


def start(args):

    # Initialize Network
    assets = AssetManager(args.prediction_dir)
    storage = PredictionStorage(args.prediction_dir)
    network = SpeechEnhancementNetwork.load(assets.get_model_cache_path(args.model_dir))
    network.start_prediction_mode()
    network.predict(np.zeros((2, 80, 24)), np.zeros((2, 128, 128, 6)))


    predicted_speech_signal = reconstruct_speech_signal\
        (AudioSignal.from_wav_file("/cs/engproj/322/real_time/raw_data/mixture.wav"), np.zeros((2, 80, 24)), 30)

    with open(assets.get_normalization_cache_path(args.model_dir), 'rb') as normalization_fd:
        video_normalizer = pickle.load(normalization_fd)

    lock = Lock()
    video_dir = assets.get_video_cache_path(args.video_audio_dir)
    predict_object = RunPredict(network, video_dir, storage.storage_dir)

    # Run video, audio, preprocess and play threads
    video_queue = Queue()
    audio_queue = Queue()
    predict_queue = Queue()
    play_queue = Queue()
    video_object = VideoProcess(video_dir)
    video_thread = Process(target=video_object.capture_frames, args=(video_queue, lock))
    audio_object = AudioProcess(assets.get_audio_cache_path(args.video_audio_dir))
    audio_thread = Process(target=audio_object.capture_frames, args=(audio_queue, lock))
    preprocess_thread = Process(target=predict_object.run_pre_process,
                                args=(video_queue, audio_queue, predict_queue, video_normalizer,
                                      lock))
    play_thread = Process(target=predict_object.play, args=(play_queue, lock))

    video_thread.start()
    audio_thread.start()
    preprocess_thread.start()
    play_thread.start()

    # Run predict
    predict_object.predict(predict_queue, play_queue, lock)

    video_thread.join()
    audio_thread.join()
    preprocess_thread.join()
    play_thread.join()

    # Save files
    predict_object.save_files(storage)

    print("*Finish All*")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-pd', '--prediction_dir', type=str, required=True)
    action_parsers = parser.add_subparsers()
    predict_parser = action_parsers.add_parser("predict")
    predict_parser.add_argument('-md', '--model_dir', type=str, required=True)
    predict_parser.add_argument('-vad', '--video_audio_dir', type=str, required=True)
    predict_parser.set_defaults(func=start)
    args = parser.parse_args()

    # For debugging
    args.func(args)

    # try:
    #     func = args.func
    #     func(args)
    # except AttributeError:
    #     parser.error("Too few arguments")


if __name__ == '__main__':
    print("Command Line Arguments:" + str(sys.argv))
    main()
