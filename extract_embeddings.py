# import the necessary packages
from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os
import glob

# this file is for changing all the frames into faces. All the detected faces are transferred here
def run_embeddings (dataset, detector):
	# dataset:				this is the folder with all the different faces. This is the big folder within small folders
	# 						of each person. This is going to be the two following persons:
	# 							- unknown
	# 							- userId
	# embeddings:			this is the file that tracks the faces
	# detector:				this is the folder of the detection models. In this file are used models
	# 						stored for the face-recognition.
	# embedding_model:		this is the path to the file for the embedding_model
	# confidence:			this is a variable that standard is on 0,5. This is not needed to change.


	# empties first the folder before adding the new files
	files = glob.glob('output/embedded_pictures/*.jpg')
	for f in files:
		os.remove(f)
	print ("Cleaning up files and start face recognizer...")


	# load our serialized face detector from disk
	# print("Loading face detector...")
	protoPath = os.path.sep.join([detector, "deploy.prototxt"])
	modelPath = os.path.sep.join([detector, "res10_300x300_ssd_iter_140000.caffemodel"])
	detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

	# load our serialized face embedding model from disk
	# print("Loading face recognizer...")
	embedder = cv2.dnn.readNetFromTorch("openface_nn4.small2.v1.t7")

	# grab the paths to the input images in our dataset
	# print("Quantifying faces...")
	imagePaths = list(paths.list_images(dataset))

	# initialize our lists of extracted facial embeddings and corresponding people names
	knownEmbeddings = []
	knownNames = []

	# initialize the total number of faces processed
	total = 0

	# loop over the image paths
	for (i, imagePath) in enumerate(imagePaths):
		# extract the person name from the image path. Commented because it is not needed to show
		# print("Processing image {}/{}".format(i + 1,len(imagePaths)))
		name = imagePath.split(os.path.sep)[-2]

		# print(imagePath)
		# load the image, resize it to have a width of 600 pixels (while maintaining the aspect ratio),
		# and then grab the image dimensions
		image = cv2.imread(imagePath) # libpng warning: iCCP: known incorrect sRGB profile

		image = imutils.resize(image, width=600)
		(h, w) = image.shape[:2]

		# construct a blob from the image
		imageBlob = cv2.dnn.blobFromImage(
			cv2.resize(image, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)

		# apply OpenCV's deep learning-based face detector to localize
		# faces in the input image
		detector.setInput(imageBlob)
		detections = detector.forward()

		# ensure at least one face was found
		if len(detections) > 0: # if detection :?
			# we're making the assumption that each image has only ONE face,  so find the bounding box
			# with the largest probability
			i = np.argmax(detections[0, 0, :, 2])
			confidence = detections[0, 0, i, 2]

			# ensure that the detection with the largest probability also
			# means our minimum probability test (thus helping filter out
			# weak detections)
			if confidence > 0.5:
				# compute the (x, y)-coordinates of the bounding box for the face
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# extract the face ROI and grab the ROI dimensions
				face = image[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]

				# save the embeddings as jpg
				number = np.random.randint(3000, size=1)
				cv2.imwrite(("output/embedded_pictures/{number}.jpg").format(number=number), face)

				# ensure the face width and height are sufficiently large
				if fW < 20 or fH < 20:
					continue

				# construct a blob for the face ROI, then pass the blob through our face embedding model
				# to obtain the 128-d quantification of the face
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
				embedder.setInput(faceBlob)
				vec = embedder.forward()

				# add the name of the person + corresponding face
				# embedding to their respective lists
				knownNames.append(name)
				knownEmbeddings.append(vec.flatten())
				total += 1

	# dump the facial embeddings + names to disk
	print("Serializing {} encodings...".format(total))
	data = {"embeddings": knownEmbeddings, "names": knownNames}
	f = open("output/embeddings.pickle", "wb")
	f.write(pickle.dumps(data))
	f.close()
