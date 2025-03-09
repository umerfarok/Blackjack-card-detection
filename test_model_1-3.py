import sys
import os
import cv2
import numpy as np
from glob import glob
from ultralytics import YOLO

# Add the SORT directory to sys.path
sys.path.append("./sort")
from sort_tracker import Sort  # SORT tracking
 
# Define paths
model_path = "./1-3_blackjack.pt"
valid_images_path = "./valid/images"
test_images_path = "./test/images"
valid_output_path = "./valid_images_inference"
test_output_path = "./test_images_inference"

# Create output directories if they don't exist
os.makedirs(valid_output_path, exist_ok=True)
os.makedirs(test_output_path, exist_ok=True)

# Load YOLO model
model = YOLO(model_path)

# Initialize SORT tracker
tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.3)

# Define classes to ignore
ignore_classes = {"wager-slot", "discard-tray", "shoe-tray", "discard-deck", "shoe-deck"}

# Function to run inference and save images
def run_inference(image_folder, output_folder):
    image_paths = glob(os.path.join(image_folder, "*.jpg")) + \
                  glob(os.path.join(image_folder, "*.png")) + \
                  glob(os.path.join(image_folder, "*.jpeg"))

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        results = model(img_path)[0]  # Get first result

        # Get detections
        boxes = results.boxes.xyxy.cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)
        scores = results.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = results.boxes.cls.cpu().numpy()  # Class IDs
        class_names = results.names  # Class name mapping

        # Filter out unwanted detections
        detections = []
        for i, class_id in enumerate(class_ids):
            class_name = class_names[int(class_id)]
            if class_name not in ignore_classes and scores[i] > 0.4:  # Confidence threshold
                detections.append([*boxes[i], scores[i]])  # Append [x1, y1, x2, y2, score]

        # Apply SORT tracking
        detections = np.array(detections)
        if len(detections) > 0:
            tracked_objects = tracker.update(detections)
        else:
            tracked_objects = []

        # Load image
        img = cv2.imread(img_path)

        # Draw bounding boxes
        for track in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, track[:5])
            label = f"Card-{track_id}"

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            
            # Add label text
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Save output image
        output_file = os.path.join(output_folder, img_name)
        cv2.imwrite(output_file, img)
        print(f"Saved: {output_file}")

# Run inference on validation and test datasets
run_inference(valid_images_path, valid_output_path)
run_inference(test_images_path, test_output_path)

print("Inference complete. Results saved!")
