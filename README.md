# ERD Diagram Analysis and Grading Pipeline

This project contains two Python scripts that form a pipeline for processing Entity-Relationship Diagrams (ERDs). The pipeline includes:
1. **Extracting text** from ERD diagram images using object detection (YOLO) and Optical Character Recognition (OCR).
2. **Using K-Nearest Neighbors (KNN)** regression to predict grades for ERD diagrams based on their extracted textual features.

---

## File Descriptions

### `text_extraction.py`
- **Purpose:** Extracts text from ERD diagram images.
- **Key Components:**
  - Uses YOLO for object detection to locate regions of interest (entities, relationships, attributes).
  - Uses EasyOCR to extract text from detected regions.
  - Saves the extracted text in `.txt` files for each input image.
- **Input:**
  - A folder containing `.png` images of ERD diagrams.
- **Output:**
  - A folder with `.txt` files, each corresponding to an input image, containing extracted text.
- **Execution Details:**
  - Automatically skips processing images that already have corresponding `.txt` files in the output folder.
  - Uses threading to process multiple images concurrently for faster execution.

### `knn_grading.py`
- **Purpose:** Predicts grades for ERD diagrams based on extracted text features.
- **Key Components:**
  - Converts the extracted text to embeddings using a pre-trained SentenceTransformer model.
  - Uses a K-Nearest Neighbors (KNN) regressor to predict grades based on similarity to labeled diagrams.
  - Outputs predicted grades for unlabeled diagrams.
- **Input:**
  - A folder containing `.txt` files with extracted text.
  - A `.csv` file (`ERD_grades.csv`) containing grades for labeled diagrams.
- **Output:**
  - A `.csv` file with predicted grades for unlabeled diagrams.
- **Execution Details:**
  - Automatically skips diagrams already present in the grade predictions.

---

## Input and Output Details

### **Text Extraction (`text_extraction.py`)**
- **Input:**
  - Folder containing `.png` ERD diagram images.
- **Output:**
  - A folder containing `.txt` files with extracted text for each processed image.

### **Grading (`knn_grading.py`)**
- **Input:**
  - Folder containing `.txt` files with extracted text.
  - `ERD_grades.csv`: A tab-separated file with columns:
    - `ERD_No`: Diagram ID.
    - `dataset1_grade` and `dataset2_grade`: Grades for labeled diagrams.
- **Output:**
  - `ERD_grades.csv`: Updated file with predicted grades for previously unlabeled diagrams.

---

## General Pipeline Details

### **Step 1: Text Extraction**
1. ERD diagram images are processed using YOLO to detect regions of interest (e.g., entities, relationships, attributes).
2. Detected regions are cropped and passed through EasyOCR to extract text.
3. Extracted text is saved in `.txt` files, one per image, in a specified output folder.

### **Step 2: Grading**
1. Extracted text is read from the `.txt` files and encoded into embeddings using a SentenceTransformer model (`all-MiniLM-L6-v2`).
2. Grades for labeled diagrams are used to train a KNN regressor.
3. Unlabeled diagrams are assigned grades based on similarity to labeled diagrams.
4. Results are saved in an updated `ERD_grades.csv` file.

---

## Requirements

### **Dependencies**
Install the required libraries before running the scripts:
```bash
pip install requrirements.txt
