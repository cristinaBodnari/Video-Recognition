from extract_embeddings import run_embeddings
from train_model import run_train_model
from build_face_dataset import camerapictures
import sys
from recognize_video import recognize_video
import shutil
import glob
import os


# this is the complete file with all the calls to the other files for the video recognition.
def run_video_recognition(userId):
    # userId:       This variable is the name of the user, normally the userId or the username


    # the camera is reversed
    try:
        camerapictures(userId, 100)
    except:
        print ('Something went wrong during camera option')
        print("4 : ", sys.exc_info())

    # now runs the face-recognition
    try:
        run_embeddings("dataset", "face_detection_model")
        run_train_model()
        recognize_video('face_detection_model', 0.85)
    except:
        print ('Could not do any face-recognition')
        print("5 : ", sys.exc_info())
    try:
        shutil.rmtree('dataset/' + userId, ignore_errors=True)
        files = glob.glob('output/embedded_pictures/*.jpg')
        for f in files:
            os.remove(f)
    except:
        print('Could not delete your folder')
        print("5 : ", sys.exc_info())


run_video_recognition('username123')
