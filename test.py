from ultralytics import YOLO
import easyocr
import numpy as np
from numpy import ndarray
import cv2

#Load trained model
model = YOLO('runs/detect/train5/weights/best.pt')  # Path to best model

image_path = 'OD_OCR_testing/OD_OCR_testing/video_games/8.png'

#Perform inference on a new image
results = model(image_path)

coord = []
labels = []
for result in results:
    result.show()
    class_ids = result.boxes.cls  # Class IDs

    labels = [model.names[int(cls)] for cls in class_ids]
    #print(labels)
    coord.append(result.boxes.xyxy)



img = cv2.imread(image_path)
picture_read = easyocr.Reader(['en'], gpu=False)


image = cv2.imread(image_path)
final = []
count = 0
for i in coord[0]:
  ci = image[int(i[1]):int(i[3]), int(i[0]):int(i[2])]
  result = picture_read.readtext(ci)
  sub = [labels[count]]
  for j in result:
    sub.append(j[1])
  if sub != []:
    final.append(sub)
  count += 1
#print(final)
for item in final:
    print(item)
