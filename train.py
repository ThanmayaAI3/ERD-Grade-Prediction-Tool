from ultralytics import YOLO

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')

# Train the model
model.train(
    data='data.yaml',  # Path to data.yaml file
    epochs=50,                      # Number of epochs
    imgsz=640,                      # Image size (YOLO will resize images to this size)
    batch=16                        # Batch size (adjust based on GPU memory)
)
