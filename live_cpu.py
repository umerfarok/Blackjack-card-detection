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
from tracker_cpu import CPUTracker  # Import the CPU-optimized tracker

print("Initializing CPU-optimized card tracking system...")

# Force CPU usage for all operations
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
torch.set_num_threads(4)  # Limit CPU threads to prevent overloading
device = 'cpu'
print(f"Using device: {device}")

# Load the YOLOv8 model with CPU optimization
try:
    # First try loading with CPU optimization
    print("Loading YOLO model optimized for CPU...")
    model = YOLO("./1-3_blackjack.pt")
    
    # Explicitly set to CPU and optimize for inference
    model.to('cpu')
    # Use smaller image size for CPU performance
    model.overrides['imgsz'] = 480  # Even smaller for CPU
    # Optimize for inference (fuse layers where possible)
    model.fuse()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Initialize our CPU-optimized tracker
tracker = CPUTracker()

# Directory to save detected images
output_dir = "detections_cpu"
os.makedirs(output_dir, exist_ok=True)

# Define valid card classes
card_classes = {
    '10c', '10d', '10h', '10s', '2c', '2d', '2h', '2s', '3c', '3d', '3h', '3s',
    '4c', '4d', '4h', '4s', '5c', '5d', '5h', '5s', '6c', '6d', '6h', '6s',
    '7c', '7d', '7h', '7s', '8c', '8d', '8h', '8s', '9c', '9d', '9h', '9s',
    'ac', 'ad', 'ah', 'as', 'jc', 'jd', 'jh', 'js', 'kc', 'kd', 'kh', 'ks',
    'qc', 'qd', 'qh', 'qs'
}

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
        # Create an OpenCV window
        cv2.namedWindow("Card Tracking (CPU)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Card Tracking (CPU)", 1024, 576)  # Smaller window for CPU
        
        # Skip more frames for better CPU performance
        frame_skip = 3  # Process every 4th frame for CPU
        frame_count = 0
        last_process_time = time.time()
        fps_avg = []
        
        container = av.open(buffer_reader, format='mp4', mode='r')
        
        for frame in container.decode(video=0):
            # Skip frames for better CPU performance
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
                
            # Calculate FPS
            current_time = time.time()
            frame_time = current_time - last_process_time
            fps = 1.0 / frame_time if frame_time > 0 else 0
            fps_avg.append(fps)
            if len(fps_avg) > 10:
                fps_avg.pop(0)
            avg_fps = sum(fps_avg) / len(fps_avg) if fps_avg else 0
            
            # Debug output every 30 frames
            if frame_count % 30 == 0:
                print(f"Processing frame {frame_count}, FPS: {avg_fps:.1f}")
                
            # Process frame
            process_frame(frame, frame_count, avg_fps)
            last_process_time = time.time()

    except Exception as e:
        print(f"[Decoder] Error: {e}")
        import traceback
        traceback.print_exc()  # Print detailed error information
    finally:
        cv2.destroyAllWindows()

def process_frame(frame, frame_idx, fps=0):
    # Decode frame to numpy array
    img = frame.to_ndarray(format='bgr24')

    if img is None:
        print(f"[ERROR] Failed to decode frame {frame_idx}")
        return
        
    # Resize for faster processing (more aggressive for CPU)
    height, width = img.shape[:2]
    scale = 480 / max(height, width)  # Scale to max dimension of 480px
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    img_proc = img.copy()

    # Run YOLO inference with CPU optimization
    try:
        # More aggressive optimization for CPU
        start_inference = time.time()
        results = model(img_proc, conf=0.3, iou=0.5, verbose=False)
        inference_time = (time.time() - start_inference) * 1000
    except Exception as e:
        print(f"[YOLO Error] {e}")
        return

    # Format detections for tracker
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            class_id = int(box.cls[0])
            label = result.names[class_id]
            
            # Only process valid card classes with sufficient confidence
            if str(label).lower() in card_classes and confidence >= 0.3:
                detections.append([x1, y1, x2, y2, confidence, label])

    # Start timing the tracker update
    start_time = time.time()
    
    # Update tracker with current frame detections
    try:
        tracked_objects = tracker.update(detections, img_proc)
    except Exception as e:
        print(f"[Tracker Error] {e}")
        import traceback
        traceback.print_exc()
        tracked_objects = []
        
    # Calculate tracking time
    tracking_time = (time.time() - start_time) * 1000  # ms
    
    # Draw performance info
    cv2.putText(img, f"FPS: {fps:.1f}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(img, f"YOLO: {inference_time:.1f}ms  Tracking: {tracking_time:.1f}ms", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Draw tracked objects with consistent IDs
    for obj in tracked_objects:
        if len(obj) < 6:  # Skip if obj doesn't have enough elements
            continue
            
        try:
            x1, y1, x2, y2, track_id, label = obj
            
            # Ensure coordinates are integers
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Check if this is a predicted detection (not directly observed)
            is_predicted = tracker.active_tracks.get(track_id, {}).get('predicted', False)
            
            # Use different color for predictions vs actual detections
            color = (0, 150, 255) if is_predicted else (0, 255, 0)  
            
            # Draw bounding box with track ID
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Display track ID and card label
            text = f"ID:{track_id} {label}"
            cv2.putText(img, text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        except Exception as e:
            print(f"[Drawing Error] {e}")

    # Save frame occasionally (every 30th processed frame)
    if frame_idx % 30 == 0:
        output_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(output_path, img)

    # Display the image
    cv2.imshow("Card Tracking (CPU)", img)
    cv2.waitKey(1)

def start_decoder():
    """ Start the decoder thread """
    threading.Thread(target=decoder_worker, daemon=True).start()

async def websocket_worker():
    """ WebSocket connection handler """
    uri = "wss://ws1.zigdpseatvkmftqo.net/BJ5.1?JSESSIONID=20AGRCD3c59DjD_gtLPM16if7ZVmv1_KhPgnNSVD4XGrjy1X4SZ5!98817579-aa178660"
    
    # Retry logic for websocket connection
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
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

                ping_task_obj = asyncio.create_task(ping_task())

                async for message in ws:
                    if isinstance(message, str):
                        print(f"[WebSocket] Received text: {message}")
                    elif isinstance(message, bytes):
                        buffer_reader.feed(message)

                ping_task_obj.cancel()
                break  # Successful connection completed
                
        except Exception as e:
            retry_count += 1
            print(f"[WebSocket] Error: {e}. Retry {retry_count}/{max_retries}")
            await asyncio.sleep(2)  # Wait before retrying
    
    if retry_count >= max_retries:
        print("[WebSocket] Failed to connect after maximum retries")

if __name__ == "__main__":
    print("Starting CPU-optimized blackjack card tracking...")
    start_decoder()
    try:
        asyncio.run(websocket_worker())
    except KeyboardInterrupt:
        print("[Main] Exiting...")
        buffer_reader.close()
        cv2.destroyAllWindows()