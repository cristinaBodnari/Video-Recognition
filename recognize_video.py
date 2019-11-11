# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import pickle
import time
import cv2
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys


# this file covers the complete video-recognition part of this model. This is the video-stream
def recognize_video(detector_, threshold_):
	# detector_:			the folder where the face-detection-models are stored
	# threshold_:			the integer for the threshold of the video recognition


	# load our serialized face detector from disk
	print("loading face detector...")
	protoPath = os.path.sep.join([detector_, "deploy.prototxt"])
	modelPath = os.path.sep.join([detector_,
								  "res10_300x300_ssd_iter_140000.caffemodel"])
	detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

	# load our serialized face embedding model from disk
	print("loading face recognizer...")
	embedder = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')

	# load the actual face recognition model along with the label encoder
	recognizer = pickle.loads(open('output/recognizer.pickle', "rb").read())
	le = pickle.loads(open('output/le.pickle', "rb").read())

	all_names = list()
	time_spend = list()
	# initialize the video stream, then allow the camera sensor to warm up
	print("starting video stream...")
	try :
		vs = VideoStream(src=0, resolution=(1920, 1080)).start()
	except :
		print("can't start video")
		print("7 : ", sys.exc_info())
		cv2.destroyAllWindows()
		vs.stop()
	time.sleep(2.0)

	# start the FPS throughput estimator
	fps = FPS().start()

	# loop over frames from the video file stream
	while True:
		# grab the frame from the threaded video stream
		frame = vs.read()

		# resize the frame to have a width of 600 pixels (while
		# maintaining the aspect ratio), and then grab the image
		# dimensions
		frame = imutils.resize(frame, width=1080)
		(h, w) = frame.shape[:2]

		# construct a blob from the image
		imageBlob = cv2.dnn.blobFromImage(
			cv2.resize(frame, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)

		# apply OpenCV's deep learning-based face detector to localize
		# faces in the input image
		detector.setInput(imageBlob)
		detections = detector.forward()

		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with
			# the prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections
			if confidence > 0.5:
				# compute the (x, y)-coordinates of the bounding box for
				# the face
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# extract the face ROI
				face = frame[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]

				# ensure the face width and height are sufficiently large
				if fW < 20 or fH < 20:
					continue

				# construct a blob for the face ROI, then pass the blob
				# through our face embedding model to obtain the 128-d
				# quantification of the face
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
												 (96, 96), (0, 0, 0), swapRB=True, crop=False)
				embedder.setInput(faceBlob)
				vec = embedder.forward()

				# perform classification to recognize the face
				preds = recognizer.predict_proba(vec)[0]
				j = np.argmax(preds)

				# the probability and the name as output of those variables
				proba = preds[j]
				name = le.classes_[j]

				# calculate the time that is spend and the names that are detected. Those are put in an array
				time_spend.append([datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
				all_names.append([name, proba])


				# draw the bounding box of the face along with the
				# associated probability
				text = "{}: {:.2f}%".format(name, proba * 100)
				y = startY - 10 if startY - 10 > 10 else startY + 10
				cv2.rectangle(frame, (startX, startY), (endX, endY),
							  (0, 0, 255), 2)
				cv2.putText(frame, text, (startX, y),
							cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

		# update the FPS counter
		fps.update()

		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# this is the end of the video stream
	# stop the timer and display FPS information
	fps.stop()

	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()


	# this will go now into the data handling of this video stream process.

	# make dataframes of the array output
	all_names = pd.DataFrame(data=all_names, columns=['name', 'probability'])
	time_spend = pd.DataFrame(data=time_spend, columns=['time'])

	# combine the two dataframes and set the threshold to delete all outputs under the threshold.
	result = pd.concat([all_names,time_spend], axis=1, join_axes=[all_names.index])
	threshold = result[result.probability >= threshold_].reset_index().drop(columns=['index'])

	# this divides the user and unknown outputs. We are going to work with the unknowns,
	# so we divide the two in two different dataframes. We name the dataframes "user" and "other"
	last_value = np.nan
	user = []
	other = []
	j = 0
	for i in threshold.name:
		if (i == last_value and i == 'unknown'):
			array = (threshold.iloc[j]).values
			other.append(array)
		elif (i == last_value and i != 'unknown'):
			array = (threshold.iloc[j]).values
			user.append(array)
		last_value = i
		j = j + 1
	other = pd.DataFrame(data=other, columns=['name', 'probability', 'time'])
	other['time'] = pd.to_datetime(other['time'])

	# this is calculating the percentage of unknown faces
	unknown_int = (threshold[threshold.name == 'unknown'].count())
	total_int = (threshold.count())
	if unknown_int[0] == 0:
		percentUnknown = "0"
	elif unknown_int[0] != 0:
		percentUnknown = (((unknown_int[0] / total_int[0])*100).round(2)).astype('str')

	# make the threshold from a number into a percentage
	threshold_percentage = (threshold_ * 100)

	# if there is no unknown face seen in the stream, this complete part will be skipped
	# if the there are any unknown faces, this process starts
	if other.empty == False:

		# this part counts the number of individual persons that are found in the screen
		# so that we can count the time of each person in the screen
		timing = []
		times = pd.DataFrame(data=timing, columns=[0, 1, 2, 'number'])
		last_value1 = datetime.now()
		k = 1
		j = 0
		try:
			for i in other.time:
				# if the last value is less or the same then 1 seconds difference, we count it as the same person
				if (i - last_value1) <= timedelta(seconds=1):
					array = (other.iloc[j]).values
					array = pd.DataFrame(data=array)
					array = array.transpose()
					array['number'] = k
					times = times.append(array, ignore_index=True)
					last_value1 = i

				# if the last value is more then 1 seconds difference, we count it as another person
				elif (i - last_value1) > timedelta(seconds=1):
					k = k + 1
					array = (other.iloc[j]).values
					array = pd.DataFrame(data=array)
					array = array.transpose()
					array['number'] = k
					times = times.append(array, ignore_index=True)
					last_value1 = i
				j = j + 1
		except:
			print ('something went wrong')

		# rename some columns of the dataframe and change the type of the time to datetime
		times.columns = ['name', 'probability', 'time', 'number']
		times['time'] = pd.to_datetime(times['time'])

		# this part counts the time that a person was in the screen. so for how long.
		# it appends the time and the number of the person to a new dataframe
		number = 1
		how_long = []
		for iterate in times:
			try:
				df = times.loc[times['number'] == number]
				unknown_in_screen = (df['time'].iloc[-1] - df['time'].iloc[0])
				how_long.append([number, unknown_in_screen])
				number = number + 1
			except:
				print (" ")
		how_long = pd.DataFrame(data=how_long, columns=['person', 'time'])
		how_long.time = pd.to_datetime(how_long.time)

		# take the number of persons by taking the last row of the column "person" to get the highest number
		persons = how_long['person'].iloc[-1]

		# this part changes the date format of each row.
		new_df = []
		for index, row in how_long.iterrows():
			try:
				row.time = (row.time.strftime("%M:%S"))
				new_df.append(row)
			except:
				print ('stop')

		# rename the columns in the dataframe and append the latest data
		how_long = pd.DataFrame(data=new_df, columns=['person', 'time'])


		# divide the time into minutes and seconds. To make in the end only a seconds column
		# this will make it easier for us to show how long persons were in the screen.
		# the time will be split, minutes to seconds and then combine the two values in to one.
		new = how_long['time'].str.split(':', n=1, expand = True)
		how_long['minute'] = new[0]
		how_long['seconds'] = new[1]
		how_long = how_long.drop(columns=['time'])
		how_long['minute'] = (how_long['minute']*60).astype('int')
		how_long['time'] = how_long['minute'] + how_long['seconds'].astype('int')
		how_long = how_long.drop(columns=['minute', 'seconds'])
		how_long = how_long.sort_values(by=['time'], ascending=False)
		sum_time = how_long['time'].sum()


		# prints all the gathered data during the video stream:
		# 	- total time
		# 	- FPS during the video
		# 	- threshold percentage (as filled in in the function)
		# 	- number of persons in the screen
		# 	- how long the longest person was in the screen
		# 	- how long the total time was of unknown persons in the screen
		# 	- the percentage unknown persons during the video stream
		print(" ")
		print("|------------------------------------------------|")
		print("   elasped time: {:.2f}".format(fps.elapsed()))
		print("   approx. FPS: {:.2f}".format(fps.fps()))
		print("   threshold >= " + str(threshold_percentage) + "%")
		print(" ")
		# print("   seen " + unknown + " times an unknown face")
		# print("   of the total of " + total + " faces")
		print("   seen " + str(persons) + " times someone else in the screen")
		print(
			"   the longest time an unknown person was for " + str(how_long['time'].iloc[0]) + " seconds in the screen")
		print("   this is a total of " + str(sum_time) + " seconds")
		print("   good to know is that " + percentUnknown + "% of the time someone else was in the screen")
		print("|------------------------------------------------|")
		print(" ")


	else:
		# this will go in when there are no unknown persons seen during the video stream. This will speed up the process
		# when everything is fine.

		# prints all the gathered data during the video stream:
		# 	- total time
		# 	- FPS during the video
		# 	- threshold percentage (as filled in in the function)
		# 	- the fact that there are no unknown persons seen
		print (" ")
		print("|------------------------------------------------|")
		print("   elasped time: {:.2f}".format(fps.elapsed()))
		print("   approx. FPS: {:.2f}".format(fps.fps()))
		print("   threshold >= "+str(threshold_percentage)+"%")
		print(" ")
		print("   no unknown persons seen during the face-recognition")
		print("|------------------------------------------------|")
		print(" ")


# recognize_video('face_detection_model', 0.85)
