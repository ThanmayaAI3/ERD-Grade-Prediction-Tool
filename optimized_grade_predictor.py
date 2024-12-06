import os
from ultralytics import YOLO
import easyocr
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define paths
input_folder = 'Collection 3 for stage 2/Collection 3/Dataset1'
output_folder = 'Dataset1'
model_path = 'runs/detect/train5/weights/best.pt'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load trained YOLO model once
model = YOLO(model_path)

# Initialize EasyOCR reader
picture_read = easyocr.Reader(['en'], gpu=True)  # Use GPU if available

def process_image(file_name):
    image_path = os.path.join(input_folder, file_name)
    output_file_name = file_name.replace('.png', '.txt')
    output_file_path = os.path.join(output_folder, output_file_name)

    # Skip processing if the text file already exists
    if os.path.exists(output_file_path):
        print(f"Skipping {file_name}: Output file already exists.")
        return

    # Perform inference
    results = model(image_path)

    coord = []
    labels = []
    for result in results:
        class_ids = result.boxes.cls  # Class IDs
        labels = [model.names[int(cls)] for cls in class_ids]
        coord.append(result.boxes.xyxy)

    # Read image and extract text
    image = cv2.imread(image_path)
    final = []
    for count, i in enumerate(coord[0]):
        ci = image[int(i[1]):int(i[3]), int(i[0]):int(i[2])]
        ocr_result = picture_read.readtext(ci)
        sub = [labels[count]]
        for j in ocr_result:
            sub.append(j[1])
        if sub:
            final.append(sub)

    # Write output to text file
    with open(output_file_path, 'w') as file:
        for item in final:
            formatted_line = str(item)  # Convert list to string format
            file.write(formatted_line + '\n')
    return output_file_name

# Use ThreadPoolExecutor for parallel processing
with ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_image, file_name)
               for file_name in os.listdir(input_folder) if file_name.endswith('.png')]

    for future in as_completed(futures):
        print(f"Processed: {future.result()}")

print(f"Processed images from '{input_folder}' and saved results to '{output_folder}'.")
