# ERD-Grade-Prediction-Tool
1. Object Detection with YOLOv8
Data Preparation: Convert the XML annotations to YOLO format, as we discussed earlier.
Model Training: Train your YOLOv8 model using the prepared dataset.
Inference: Use the trained model to detect ERD components in new images.
2. Text Extraction with EasyOCR
Coordinate Extraction: Pass the coordinates of the detected objects to EasyOCR.
Text Recognition: Extract text from the specified regions in the image.
Output: Return the object type along with the recognized text.
