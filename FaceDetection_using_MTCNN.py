import mtcnn
from mtcnn.mtcnn import MTCNN
from PIL import Image
from numpy import asarray
import matplotlib.pyplot as plt
import cv2

# extract a single face from a given photograph
def extract_face(filename, required_size=(48, 48)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
  
    faceDetected = []
    for i in range(len(results)):
	    print(results[i]['box'],results[i]['confidence'],i)
    
	    x1, y1, width, height = results[i]['box']

	    x1, y1 = abs(x1), abs(y1)
	    x2, y2 = x1 + width, y1 + height
	    # extract the face
	    face = pixels[y1:y2, x1:x2]
	    # resize pixels to the model size
	    image = Image.fromarray(face)
	    image = image.resize(required_size)
	    face_array = asarray(image)
	    faceDetected.append(face_array)
    return faceDetected


# load the photo and extract the face
pixels = extract_face('Input Image Path')
count = 0
for i in pixels:
	count+=1
	j = cv2.cvtColor(i,cv2.COLOR_BGR2RGB)
	cv2.imshow('f',j)
	cv2.waitKey(0)
	#Image can have more than one face,so this will write the image name as count_no. of face
	cv2.imwrite('Output Image Path.png{}'.format(count),j)
	
