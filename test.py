from ultralytics import YOLO

# Load your trained model
model = YOLO('runs/detect/train5/weights/best.pt')  # Path to your best model

# Perform inference on a new image
results = model('dataset/images/val/123.png')

for result in results:
    result.show()