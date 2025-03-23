import asyncio
import json
import threading
import io
import time
import os
import cv2
import av
import websockets
import numpy as np
from ultralytics import YOLO
import torch
from tracker import Tracker  # Import our enhanced tracker

# Check CUDA availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the fine-tuned YOLO model with CUDA if available
# model = YOLO("./1-3_blackjack.pt")
model = YOLO("../models-yolo/best_23.pt")
model.to(device)  # Move model to available device

# GPU optimization settings
if device == 'cuda':
    # Enable TensorRT acceleration if available
    model.fuse()  # Fuse model layers for better performance
    # Set batch size to process more frames at once
    batch_size = 4
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
    torch.backends.cudnn.deterministic = False  # Better performance but less deterministic
    print(f"CUDA optimization enabled. Using batch size: {batch_size}")
    # Check available GPU memory
    torch_mem = torch.cuda.get_device_properties(0).total_memory / 1e9  # Convert to GB
    print(f"Total GPU memory: {torch_mem:.2f} GB")
else:
    batch_size = 1

# Initialize our enhanced tracker
tracker = Tracker()

# Directory to save detected images
output_dir = "detections_1"
os.makedirs(output_dir, exist_ok=True)

# Define valid card classes (original format)
card_classes = {
    '10c', '10d', '10h', '10s', '2c', '2d', '2h', '2s', '3c', '3d', '3h', '3s',
    '4c', '4d', '4h', '4s', '5c', '5d', '5h', '5s', '6c', '6d', '6h', '6s',
    '7c', '7d', '7h', '7s', '8c', '8d', '8h', '8s', '9c', '9d', '9h', '9s',
    'ac', 'ad', 'ah', 'as', 'jc', 'jd', 'jh', 'js', 'kc', 'kd', 'kh', 'ks',
    'qc', 'qd', 'qh', 'qs'
}

# Add debugging flag
DEBUG_LABELS = True
# Create a dictionary to store detected class names for debugging
detected_classes = {}

# Helper function to normalize card labels between different formats
def normalize_card_label(label):
    """
    Normalize card labels to a consistent format.
    Can handle various input formats like '10c', '10_of_clubs', etc.
    """
    label = str(label).lower()
    
    # If already in our short format (e.g., '10c', 'as')
    if label in card_classes:
        return label
        
    # Handle potential format like '10_of_clubs', 'ace_of_spades', etc.
    card_value_map = {
        'ace': 'a', 'king': 'k', 'queen': 'q', 'jack': 'j',
        'a': 'a', 'k': 'k', 'q': 'q', 'j': 'j',
        '10': '10', '9': '9', '8': '8', '7': '7', '6': '6', 
        '5': '5', '4': '4', '3': '3', '2': '2'
    }
    
    suit_map = {
        'clubs': 'c', 'diamonds': 'd', 'hearts': 'h', 'spades': 's',
        'club': 'c', 'diamond': 'd', 'heart': 'h', 'spade': 's',
        'c': 'c', 'd': 'd', 'h': 'h', 's': 's'
    }
    
    # Try to extract value and suit from various formats
    for val, short_val in card_value_map.items():
        if val in label:
            for suit, short_suit in suit_map.items():
                if suit in label:
                    return f"{short_val}{short_suit}"
    
    # If we couldn't parse it, return original label
    return label

class BufferReader(io.RawIOBase):
    def __init__(self):
        self.buffer = bytearray()
        self.closed_flag = False
        self.lock = threading.Lock()
        self.data_available = threading.Condition(self.lock)

    def feed(self, data: bytes):
        with self.lock:
            self.buffer.extend(data)
            self.data_available.notify_all()

    def read(self, n=-1):
        with self.lock:
            while not self.buffer and not self.closed_flag:
                self.data_available.wait()
            if not self.buffer:
                return b''
            if n < 0 or n > len(self.buffer):
                n = len(self.buffer)
            result = bytes(self.buffer[:n])
            del self.buffer[:n]
            return result

    def readable(self):
        return True

    def close(self):
        with self.lock:
            self.closed_flag = True
            self.data_available.notify_all()
        super().close()

buffer_reader = BufferReader()

def decoder_worker():
    """ Decodes video stream and runs YOLO inference with tracking """
    try:
        # Create an OpenCV window instead of using matplotlib
        cv2.namedWindow("Card Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Card Tracking", 1280, 720)
        
        container = av.open(buffer_reader, format='mp4', mode='r')
        
        # Enable CUDA stream if available
        stream = torch.cuda.Stream() if device == 'cuda' else None
        
        for frame_idx, frame in enumerate(container.decode(video=0)):
            if stream:
                with torch.cuda.stream(stream):
                    process_frame(frame, frame_idx)
            else:
                process_frame(frame, frame_idx)

    except Exception as e:
        print(f"[Decoder] Error: {e}")
        import traceback
        traceback.print_exc()  # Print detailed error information
    finally:
        cv2.destroyAllWindows()
        if device == 'cuda':
            torch.cuda.empty_cache()

def process_frame(frame, frame_idx):
    img = frame.to_ndarray(format='bgr24')

    if img is None:
        print(f"[ERROR] Failed to decode frame {frame_idx}")
        return

    # Make a copy for processing
    img_proc = img.copy()

    # Run YOLO inference with optimized settings
    try:
        # Use half precision for faster inference
        if device == 'cuda':
            results = model(img_proc, conf=0.25, iou=0.45, verbose=False, half=True)
        else:
            results = model(img_proc, conf=0.25, iou=0.45, verbose=False)
    except Exception as e:
        print(f"[YOLO Error] {e}")
        return

    # Debug: Print out class names to understand model output format
    if DEBUG_LABELS and frame_idx % 30 == 0:  # Only print every 30 frames to avoid flooding console
        print("\n--- Detected Classes Debug Info ---")
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = box.conf[0].item()
                label = result.names[class_id]
                # Store unique class names in our dictionary
                if label not in detected_classes:
                    detected_classes[label] = confidence
                    print(f"New class detected: '{label}' with confidence {confidence:.2f}")
        print(f"All detected classes so far: {list(detected_classes.keys())}")
        print("-----------------------------------\n")

    # Format detections for tracker
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            class_id = int(box.cls[0])
            label = result.names[class_id]
            
            # Normalize the label to our expected format
            normalized_label = normalize_card_label(label)
            
            # Check both original label and normalized label
            if normalized_label in card_classes and confidence >= 0.25:
                detections.append([x1, y1, x2, y2, confidence, normalized_label])

    # Update tracker with current frame detections
    try:
        tracked_objects = tracker.update(detections, img_proc)
    except Exception as e:
        print(f"[Tracker Error] {e}")
        import traceback
        traceback.print_exc()
        tracked_objects = []

    # Draw tracked objects with consistent IDs
    for obj in tracked_objects:
        if len(obj) < 6:  # Skip if obj doesn't have enough elements
            continue
            
        try:
            x1, y1, x2, y2, track_id, label = obj
            
            # Ensure coordinates are integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Draw bounding box with track ID
            color = (0, 255, 0)  # Green for tracked objects
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Display track ID and card label
            text = f"ID:{track_id} {label}"
            cv2.putText(img, text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except Exception as e:
            print(f"[Drawing Error] {e}")

    # Save detected frame
    output_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
    cv2.imwrite(output_path, img)

    # Display the image using OpenCV
    cv2.imshow("Card Tracking", img)
    cv2.waitKey(1)

def start_decoder():
    """ Start the decoder thread """
    threading.Thread(target=decoder_worker, daemon=True).start()

async def websocket_worker():
    """ WebSocket connection handler """
    uri = "Live- video Url"
    
    try:
        async with websockets.connect(uri) as ws:
            print("[WebSocket] Connected.")
            await ws.send("iq0")

            async def ping_task():
                """ Sends periodic ping messages to keep the connection alive """
                counter = 1 
                while True:
                    await asyncio.sleep(30)
                    ping_msg = {
                        "cmd": "ping",
                        "counter": counter,
                        "clientTime": int(time.time() * 1000)
                    }
                    await ws.send(json.dumps(ping_msg))
                    counter += 1

            asyncio.create_task(ping_task())

            async for message in ws:
                if isinstance(message, str):
                    print(f"[WebSocket] Received text: {message}")
                elif isinstance(message, bytes):
                    buffer_reader.feed(message)

    except Exception as e:
        print(f"[WebSocket] Error: {e}")

if __name__ == "__main__":
    start_decoder()
    try:
        asyncio.run(websocket_worker())
    except KeyboardInterrupt:
        print("[Main] Exiting...")
        buffer_reader.close()
        cv2.destroyAllWindows()
