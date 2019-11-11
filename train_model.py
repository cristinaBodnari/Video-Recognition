from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle


def run_train_model():
	# embeddings_:			the path that is been used for the embedded pictures
	# recognizer_:			the path that is used for the recognition
	# le_:					the path for the le.pickle file


	# load the face embeddings
	print("Train model...")
	data = pickle.loads(open("output/embeddings.pickle", "rb").read())

	# encode the labels
	# print("Encoding labels...")
	le = LabelEncoder()
	labels = le.fit_transform(data["names"])

	# train the model used to accept the 128-d embeddings of the face and then produce the actual face recognition
	# print("Training model...")
	recognizer = SVC(C=1.0, kernel="linear", probability=True)
	recognizer.fit(data["embeddings"], labels)

	# write the actual face recognition model to disk
	f = open("output/recognizer.pickle", "wb")
	f.write(pickle.dumps(recognizer))
	f.close()

	# write the label encoder to disk
	f = open("output/le.pickle", "wb")
	f.write(pickle.dumps(le))
	f.close()
