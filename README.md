## Project: ERD Diagram Analysis and Grading Pipeline
This project contains two Python scripts that form a pipeline for processing Entity-Relationship Diagrams (ERDs). The pipeline includes:

Extracting text from ERD diagram images using object detection (YOLO) and Optical Character Recognition (OCR).
Using K-Nearest Neighbors (KNN) regression to predict grades for ERD diagrams based on their extracted textual features.
1. File Descriptions
File 1: text_extraction.py
Purpose: Extracts text from ERD diagram images.
Key Components:
Uses YOLO for object detection to locate regions of interest (entities, relationships, attributes).
Uses EasyOCR to extract text from detected regions.
Saves the extracted text in .txt files for each input image.
Input:
A folder containing .png images of ERD diagrams.
Output:
A folder with .txt files, each corresponding to an input image, containing extracted text.
Execution:
Automatically skips processing images that already have corresponding .txt files in the output folder.
Uses threading to speed up processing by handling multiple images concurrently.
File 2: knn_grading.py
Purpose: Predicts grades for ERD diagrams based on extracted text features.
Key Components:
Converts the extracted text to embeddings using a pre-trained SentenceTransformer model.
Uses a K-Nearest Neighbors (KNN) regressor to predict grades based on similarity to labeled diagrams.
Outputs predicted grades for unlabeled diagrams.
Input:
A folder containing .txt files with extracted text.
A .csv file (ERD_grades.csv) containing grades for labeled diagrams.
Output:
A .csv file with predicted grades for unlabeled diagrams.
Execution:
Automatically skips diagrams already present in the grade predictions.
