from imutils.video import VideoStream
import imutils
import time
import cv2
import os
import sys


def camerapictures(userId, numberOfPictures):
	# userId:			this is the name of the folder. This is or a username or a userId
	# numberOfPictures:	this variable indicates the number of pictures that is used for the dataset


	# load OpenCV's Haar cascade for face detection from disk
	detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
	if not os.path.exists('dataset/' + userId):
		os.makedirs('dataset/' + userId)

	# initialize the video stream, allow the camera sensor to warm up,
	# and initialize the total number of example faces written to disk
	# thus far
	try :
		print("Starting picture stream...")
		vs = VideoStream(src=0, resolution=(1920,1080)).start()
	except :
		print("can't start video")
		print("6 : ", sys.exc_info())
		cv2.destroyAllWindows()
		vs.stop()
	# vs = VideoStream(usePiCamera=True).start()
	time.sleep(2.0)
	total = 0
	variable = 1
	# loop over the frames from the video stream
	while variable <= numberOfPictures:
		# grab the frame from the threaded video stream, clone it, (just in case we want to write it to disk),
		# and then resize the frame so we can apply face detection faster
		frame = vs.read()
		orig = frame.copy()
		frame = imutils.resize(frame, width=1080)

		# detect faces in the grayscale frame
		rects = detector.detectMultiScale(
			cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
			minNeighbors=5, minSize=(30, 30))

		# loop over the face detections and draw them on the frame
		for (x, y, w, h) in rects:
			cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# 150 images are automatically written on the disk, the *original* picture
		# so we can later process it and use it for face recognition
		if variable <= numberOfPictures:
			p = os.path.sep.join(["dataset/"+userId, "{}.jpg".format(
				str(total).zfill(5))])
			cv2.imwrite(p, orig)
			variable += 1
			total += 1

		# if the `q` key was pressed, break from the loop
		elif key == ord("q") or variable >= numberOfPictures:
			break

	# do a bit of cleanup
	print("{} face images stored".format(total))
	print("Cleaning up...")
	cv2.destroyAllWindows()
	# vs.release() # c'est quoi ça ???
	vs.stop()
