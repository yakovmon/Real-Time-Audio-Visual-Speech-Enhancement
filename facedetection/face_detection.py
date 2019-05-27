import os

import numpy as np
import dlib
import cv2


class FaceDetector:

	FACIAL_LANDMARK_IDS = range(68)
	MOUTH_LANDMARK_IDS = range(48, 68)

	def __init__(self):
		self._detector = dlib.get_frontal_face_detector()
		self._landmark_predictor = dlib.shape_predictor(
			os.path.join(os.path.dirname(__file__), "shape_predictor_68_face_landmarks.dat")
		)

	def detect_face(self, image, bounding_box_shape=None):
		detection = self._detect_face(image)

		bounding_box = BoundingBox(detection.left(), detection.top(), detection.right(), detection.bottom())
		if bounding_box_shape is not None:
			bounding_box.resize_equally(*bounding_box_shape)

		return bounding_box

	def crop_face(self, image, bounding_box_shape=None):
		bounding_box = self.detect_face(image, bounding_box_shape)

		return self._crop(image, bounding_box)

	def detect_mouth(self, image, bounding_box_shape=None):
		landmarks = self._detect_landmarks(image, FaceDetector.MOUTH_LANDMARK_IDS)

		points = np.array([(landmark.x, landmark.y) for landmark in landmarks], dtype=np.float32)
		(mouth_x, mouth_y, mouth_width, mouth_height) = cv2.boundingRect(points)

		bounding_box = BoundingBox(mouth_x, mouth_y, mouth_x + mouth_width - 1, mouth_y + mouth_height - 1)
		if bounding_box_shape is not None:
			bounding_box.resize_equally(*bounding_box_shape)

		return bounding_box

	def crop_mouth(self, image, bounding_box_shape=None):
		bounding_box = self.detect_mouth(image, bounding_box_shape)

		return self._crop(image, bounding_box)

	def _detect_landmarks(self, image, landmark_ids):
		detection = self._detect_face(image)
		landmarks = self._landmark_predictor(image, detection)

		return [Landmark(landmarks.part(i).x, landmarks.part(i).y) for i in landmark_ids]

	def _detect_face(self, image):
		detections, scores, idx = self._detector.run(image, upsample_num_times=0, adjust_threshold=-1)

		if len(detections) == 0:
			raise Exception("frame contains 0 faces")

		if len(detections) == 1:
			return detections[0]

		best_detection = np.argmax(scores)
		return detections[best_detection]

	@staticmethod
	def _crop(image, bounding_box):
		if bounding_box.top < 0 or bounding_box.bottom >= image.shape[0] \
			or bounding_box.left < 0 or bounding_box.right >= image.shape[1]:

			raise Exception("bounding box exceeds image dimensions")

		return image[
			bounding_box.top: bounding_box.top + bounding_box.get_height(),
			bounding_box.left: bounding_box.left + bounding_box.get_width()
		]


class BoundingBox:

	def __init__(self, left, top, right, bottom):
		self.left = left
		self.top = top
		self.right = right
		self.bottom = bottom

	def get_width(self):
		return self.right - self.left + 1

	def get_height(self):
		return self.bottom - self.top + 1

	def resize_equally(self, width, height):
		extra_width = width - self.get_width()
		self.left -= int(extra_width / 2)
		self.right += extra_width - int(extra_width / 2)

		extra_height = height - self.get_height()
		self.top -= int(extra_height / 2)
		self.bottom += extra_height - int(extra_height / 2)


class Landmark:

	def __init__(self, x, y):
		self.x = x
		self.y = y
