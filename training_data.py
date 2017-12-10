import numpy as np
from PIL import Image
import os
import cv2


#list to hold all subject faces
faces = []
labels = []

data_folder_path = "/home/vishal/BTP_Final/training_data/"
face_recognizer = cv2.face.createLBPHFaceRecognizer()

dirs = os.listdir(data_folder_path)

for dir_name in dirs:

	if not dir_name.startswith("s"):
		continue;

	label = int(dir_name.replace("s", ""))

	subject_dir_path = data_folder_path + "/" + dir_name

	subject_images_names = os.listdir(subject_dir_path)

	for image_name in subject_images_names:

		if image_name.startswith("."):
			continue;

		image_path = subject_dir_path + "/" + image_name

		image = cv2.imread(image_path)
		
		image=image.convert('L')

		cv2.imshow("Training on image...", image)
		cv2.waitKey(100)

		face = image

		if face is not None:
			faces.append(face)
			labels.append(label)

face_recognizer.train(faces, np.array(labels))

cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()
