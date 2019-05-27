import imageio
import cv2
import numpy as np


class VideoFileReader:

    def __init__(self, video_file_path):
        self._video_fd = imageio.get_reader(video_file_path)

    def close(self):
        self._video_fd.close()

    def read_all_frames(self, convert_to_gray_scale=False):
        if convert_to_gray_scale:
            video_shape = (self.get_frame_count(), self.get_frame_height(), self.get_frame_width())
        else:
            video_shape = (self.get_frame_count(), self.get_frame_height(), self.get_frame_width(), 3)

        frames = np.ndarray(shape=video_shape, dtype=np.uint8)
        for i in range(self.get_frame_count()):
            frames[i, ] = self.read_next_frame(convert_to_gray_scale=convert_to_gray_scale)

        return frames

    def read_next_frame(self, convert_to_gray_scale=False):
        frame = self._video_fd.get_next_data()

        if convert_to_gray_scale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame

    def get_frame_rate(self):
        return self._video_fd.get_meta_data()["fps"]

    def get_frame_size(self):
        return self._video_fd.get_meta_data()["size"]

    def get_frame_count(self):
        return self._video_fd.get_length()

    def get_frame_width(self):
        return self.get_frame_size()[0]

    def get_frame_height(self):
        return self.get_frame_size()[1]

    def get_format(self):
        return dict(
            frame_rate=self.get_frame_rate(),
            frame_width=self.get_frame_width(),
            frame_height=self.get_frame_height()
        )

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()


class VideoFileWriter:

    def __init__(self, video_file_path, frame_rate):
        self._video_fd = imageio.get_writer(video_file_path, fps=frame_rate)

    def close(self):
        self._video_fd.close()

    def write_frame(self, frame):
        self._video_fd.append_data(frame)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()
