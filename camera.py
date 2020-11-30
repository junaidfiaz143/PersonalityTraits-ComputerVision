# from PIL import ImageFont, ImageDraw, Image
from collections import OrderedDict
import numpy as np
import dlib
import cv2

class VideoCamera(object):
	def __init__(self):
		# self.font = ImageFont.truetype("fonts/clan_med.ttf", 10)  

		self.detector = dlib.get_frontal_face_detector() #(HOG-based)
		self.predictor = dlib.shape_predictor("predictor/shape_predictor_68_face_landmarks.dat")

		self.FACIAL_LANDMARKS_IDXS = OrderedDict([
			("mouth", (48, 68)),
			("inner_mouth", (60, 68)),
			("right_eyebrow", (17, 22)),
			("left_eyebrow", (22, 27)),
			("right_eye", (36, 42)),
			("left_eye", (42, 48)),
			("nose", (27, 36)),
			("jaw", (0, 17))
		])

		(self.lStart, self.lEnd) = self.FACIAL_LANDMARKS_IDXS["left_eye"]
		(self.rStart, self.rEnd) = self.FACIAL_LANDMARKS_IDXS["right_eye"]

		(self.mStart, self.mEnd) = self.FACIAL_LANDMARKS_IDXS["mouth"]

		(self.nStart, self.nEnd) = self.FACIAL_LANDMARKS_IDXS["nose"]

		self.draw_face = False

		self.video = cv2.VideoCapture(0)
		# self.video = cv2.VideoCapture("school.mp4")

	def __del__(self):
		self.video.release()

	def rect_to_bb(rect):
		x = rect.left()
		y = rect.top()
		w = rect.right() - x
		h = rect.bottom() - y

	def shape_to_np(self, shape, dtype="int"):
		coords = np.zeros((shape.num_parts, 2), dtype=dtype)

		for i in range(0, shape.num_parts):
			coords[i] = (shape.part(i).x, shape.part(i).y)

		return coords

	def euclidean_dist(self, ptA, ptB):
		return np.sqrt(np.sum((ptA - ptB)**2))

	def eye_aspect_ratio(self, eye):
		A = self.euclidean_dist(eye[1], eye[5])
		B = self.euclidean_dist(eye[2], eye[4])
		C = self.euclidean_dist(eye[0], eye[3])

		aspect_ration = (A + B) / (2.0 * C)
		return aspect_ration

	def get_frame(self):
		success, frame = self.video.read()

		properties = []

		frame = cv2.resize(frame, (500,500), interpolation=cv2.INTER_AREA)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		rects = self.detector(gray, 1)

		for (i, rect) in enumerate(rects):

			shape = self.predictor(gray, rect)

			shape = self.shape_to_np(shape)

			leftEye = shape[self.lStart:self.lEnd]
			rightEye = shape[self.rStart:self.rEnd]
			leftEAR = self.eye_aspect_ratio(leftEye)
			rightEAR = self.eye_aspect_ratio(rightEye)

			eyeRatio = (leftEAR + rightEAR) / 2.0

			if eyeRatio > 0.33:
				# print("EYE - OPEN: ", eyeRatio)
				properties.append("Angry / Serious / Moody")
			else:
				# print("EYE - CLOSE: ", eyeRatio)
				properties.append("Relaxed")

			######################################### 

			mouth = shape[self.mStart:self.mEnd]

			mouthHull = cv2.convexHull(mouth)
			cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

			mouthRatio = self.euclidean_dist(shape[62], shape[66])

			if mouthRatio > 3.0:
				# print("MOUTH - OPEN: ", mouthRatio)
				properties.append("Scold / Passionate")
			else:
				# print("MOUTH - CLOSE: ", mouthRatio)
				properties.append("Quite / Shy")

			#########################################

			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

			nose = shape[self.nStart:self.nEnd]
			noseHull = cv2.convexHull(nose)
			cv2.drawContours(frame, [noseHull], -1, (0, 255, 0), 1)	
			noseRatio = self.euclidean_dist(shape[31], shape[35])

			print("NOSE: ", noseRatio)
			if noseRatio > 45.0:
				properties.append("Greedy / Good Planner")
			else:
				properties.append("Quite Wiser")
			print("=-=-=-=-=-=-=-=-=-=-=")

			if self.draw_face:
				(x, y, w, h) = self.rect_to_bb(rect)
				cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

				cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

				for (x, y) in shape:
					cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

		# frame = Image.fromarray(frame)


		frame = cv2.flip(frame, 1)
		ret, jpeg = cv2.imencode('.png', frame)
		return jpeg.tobytes(), properties