from PIL import ImageFont, ImageDraw, Image
from collections import OrderedDict
import numpy as np
import dlib
import cv2

font = ImageFont.truetype("fonts/clan_med.ttf", 10)  

detector = dlib.get_frontal_face_detector() #(HOG)
predictor = dlib.shape_predictor("predictor/shape_predictor_68_face_landmarks.dat")

def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

def shape_to_np(shape, dtype="int"):
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)

	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	return coords

def euclidean_dist(ptA, ptB):
	return np.sqrt(np.sum((ptA - ptB)**2))

def eye_aspect_ratio(eye):
	A = euclidean_dist(eye[1], eye[5])
	B = euclidean_dist(eye[2], eye[4])
	C = euclidean_dist(eye[0], eye[3])

	aspect_ration = (A + B) / (2.0 * C)
	return aspect_ration

FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("inner_mouth", (60, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])

(lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]

(mStart, mEnd) = FACIAL_LANDMARKS_IDXS["inner_mouth"]

draw_face = False

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while(True):

	ret, frame = video.read()

	if ret == False:
		break

	frame = cv2.resize(frame, (480,320), interpolation=cv2.INTER_AREA)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	rects = detector(gray, 1)

	for (i, rect) in enumerate(rects):

		shape = predictor(gray, rect)

		shape = shape_to_np(shape)

		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		aspect_ratio = (leftEAR + rightEAR) / 2.0

		if aspect_ratio > 0.25:
			print("EYE - OPEN: ", aspect_ratio)
		elif aspect_ratio <= 0.25:
			print("EYE - CLOSE: ", aspect_ratio)

		mouth = shape[mStart:mEnd]

		mouthHull = cv2.convexHull(mouth)
		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

		mouthRatio = euclidean_dist(shape[62], shape[66])

		if mouthRatio > 3.0:
			print("MOUTH - OPEN: ", mouthRatio)
		elif mouthRatio <= 3.0:
			print("MOUTH - CLOSE: ", mouthRatio)

		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		nose = shape[31:36]
		noseHull = cv2.convexHull(nose)
		cv2.drawContours(frame, [noseHull], -1, (0, 255, 0), 1)	
		noseRatio = euclidean_dist(shape[31], shape[35])

		print("NOSE: ", noseRatio)
		print("=-=-=-=-=-=-=-=-=-=-=")

		if draw_face:
			(x, y, w, h) = rect_to_bb(rect)
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

			cv2.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

			for (x, y) in shape:
				cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

	frame = Image.fromarray(frame)

	draw = ImageDraw.Draw(frame)
	draw.rectangle([(0, 0), (100, 40)] , width=1, outline=(255, 255, 255), fill=(0,0,255))
	draw.text((10, 15), "Press Q to exit.", font=font, fill=(255, 255, 255))

	cv2.imshow("Facial Featues", np.array(frame))

	if cv2.waitKey(1) == ord('q'):
		break

video.release()
cv2.destroyAllWindows()