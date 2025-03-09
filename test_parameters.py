#!/usr/bin/env python3
"""
Parameter testing script for blackjack card tracking system.
This script allows you to test different tracking parameters on a video stream.
"""

import argparse
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

# Buffer reader for video stream
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

# Parameters to test
class Parameters:
    def __init__(self, 
                 use_gpu=False,
                 max_age=30, 
                 n_init=3, 
                 max_cosine_distance=0.5,
                 max_iou_distance=0.7,
                 detection_confidence=0.25,
                 nms_iou_threshold=0.45,
                 reid_similarity=0.75,
                 frame_skip=1):
        self.use_gpu = use_gpu
        self.max_age = max_age
        self.n_init = n_init
        self.max_cosine_distance = max_cosine_distance
        self.max_iou_distance = max_iou_distance
        self.detection_confidence = detection_confidence
        self.nms_iou_threshold = nms_iou_threshold
        self.reid_similarity = reid_similarity
        self.frame_skip = frame_skip
        
    def __str__(self):
        return (f"Parameters:\n"
                f"  Use GPU: {self.use_gpu}\n"
                f"  Max Age: {self.max_age}\n"
                f"  N Init: {self.n_init}\n"
                f"  Max Cosine Distance: {self.max_cosine_distance}\n"
                f"  Max IoU Distance: {self.max_iou_distance}\n"
                f"  Detection Confidence: {self.detection_confidence}\n"
                f"  NMS IoU Threshold: {self.nms_iou_threshold}\n"
                f"  ReID Similarity: {self.reid_similarity}\n"
                f"  Frame Skip: {self.frame_skip}")

# Main function
def main():
    parser = argparse.ArgumentParser(description="Test different tracking parameters")
    
    # Tracker parameters
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    parser.add_argument("--max-age", type=int, default=30, help="Max age for track")
    parser.add_argument("--n-init", type=int, default=3, help="Frames needed to confirm a track")
    parser.add_argument("--cosine-dist", type=float, default=0.5, help="Max cosine distance")
    parser.add_argument("--iou-dist", type=float, default=0.7, help="Max IoU distance")
    
    # Detection parameters
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--reid-sim", type=float, default=0.75, help="ReID similarity threshold")
    
    # Performance parameters
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame")
    
    args = parser.parse_args()
    
    # Set parameters
    params = Parameters(
        use_gpu=args.gpu,
        max_age=args.max_age,
        n_init=args.n_init,
        max_cosine_distance=args.cosine_dist,
        max_iou_distance=args.iou_dist,
        detection_confidence=args.conf,
        nms_iou_threshold=args.iou,
        reid_similarity=args.reid_sim,
        frame_skip=args.frame_skip
    )
    
    print(params)
    
    # Create output directory
    output_dir = f"test_results/test_{int(time.time())}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save parameters to file
    with open(os.path.join(output_dir, "parameters.txt"), "w") as f:
        f.write(str(params))
    
    # Choose correct imports based on GPU/CPU setting
    if params.use_gpu:
        print("Using GPU mode")
        from tracker import Tracker
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        print("Using CPU mode")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
        from tracker_cpu import CPUTracker as Tracker
        device = 'cpu'
        
    print(f"Using device: {device}")
    
    # Load YOLO model
    try:
        print("Loading YOLO model...")
        model = YOLO("./1-3_blackjack.pt")
        model.to(device)
        
        if not params.use_gpu:
            # Optimize for CPU
            model.overrides['imgsz'] = 480
        
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create tracker with custom parameters
    if params.use_gpu:
        tracker = Tracker()
        # Override tracker parameters
        tracker.tracker = tracker.tracker.__class__(
            max_age=params.max_age,
            n_init=params.n_init,
            max_cosine_distance=params.max_cosine_distance,
            max_iou_distance=params.max_iou_distance
        )
        # Override ReID parameters
        tracker.reid_model.similarity_threshold = params.reid_similarity
    else:
        tracker = Tracker()
        # Override tracker parameters
        tracker.tracker = tracker.tracker.__class__(
            max_age=params.max_age,
            n_init=params.n_init,
            max_cosine_distance=params.max_cosine_distance,
            max_iou_distance=params.max_iou_distance
        )
        # Override ReID parameters
        tracker.reid_model.similarity_threshold = params.reid_similarity
    
    # Define valid card classes
    card_classes = {
        '10c', '10d', '10h', '10s', '2c', '2d', '2h', '2s', '3c', '3d', '3h', '3s',
        '4c', '4d', '4h', '4s', '5c', '5d', '5h', '5s', '6c', '6d', '6h', '6s',
        '7c', '7d', '7h', '7s', '8c', '8d', '8h', '8s', '9c', '9d', '9h', '9s',
        'ac', 'ad', 'ah', 'as', 'jc', 'jd', 'jh', 'js', 'kc', 'kd', 'kh', 'ks',
        'qc', 'qd', 'qh', 'qs'
    }
    
    # Create buffer reader for video stream
    buffer_reader = BufferReader()
    
    def process_frame(frame, frame_idx):
        img = frame.to_ndarray(format='bgr24')
        if img is None:
            print(f"[ERROR] Failed to decode frame {frame_idx}")
            return
        
        # Resize for CPU mode
        if not params.use_gpu:
            height, width = img.shape[:2]
            scale = 480 / max(height, width)
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        
        img_proc = img.copy()
        
        # Run YOLO inference
        try:
            start_inference = time.time()
            results = model(img_proc, 
                           conf=params.detection_confidence, 
                           iou=params.nms_iou_threshold, 
                           verbose=False)
            inference_time = (time.time() - start_inference) * 1000
        except Exception as e:
            print(f"[YOLO Error] {e}")
            return
            
        # Format detections
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0].item()
                class_id = int(box.cls[0])
                label = result.names[class_id]
                
                # Only process valid card classes
                if str(label).lower() in card_classes and confidence >= params.detection_confidence:
                    detections.append([x1, y1, x2, y2, confidence, label])
                    
        # Track objects
        start_time = time.time()
        tracked_objects = tracker.update(detections, img_proc)
        tracking_time = (time.time() - start_time) * 1000
        
        # Draw performance info
        cv2.putText(img, f"Frame: {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(img, f"YOLO: {inference_time:.1f}ms  Tracking: {tracking_time:.1f}ms", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                   
        # Draw bounding boxes
        for obj in tracked_objects:
            if len(obj) < 6:
                continue
                
            x1, y1, x2, y2, track_id, label = obj
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Check if this is a predicted detection
            is_predicted = False
            if hasattr(tracker, 'active_tracks'):
                is_predicted = tracker.active_tracks.get(track_id, {}).get('predicted', False)
            
            color = (0, 150, 255) if is_predicted else (0, 255, 0)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            text = f"ID:{track_id} {label}"
            cv2.putText(img, text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                       
        # Save frame
        if frame_idx % 30 == 0:
            output_path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(output_path, img)
            
        # Display image
        cv2.imshow("Parameter Testing", img)
        cv2.waitKey(1)
        
    def decoder_worker():
        try:
            # Create window
            cv2.namedWindow("Parameter Testing", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Parameter Testing", 1280, 720)
            
            container = av.open(buffer_reader, format='mp4', mode='r')
            
            frame_count = 0
            for frame in container.decode(video=0):
                frame_count += 1
                
                # Apply frame skip
                if frame_count % params.frame_skip != 0:
                    continue
                    
                process_frame(frame, frame_count)
                
        except Exception as e:
            print(f"[Decoder] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cv2.destroyAllWindows()
            
    # Start decoder thread
    threading.Thread(target=decoder_worker, daemon=True).start()
    
    # WebSocket function
    async def websocket_worker():
        uri = "wss://ws1.zigdpseatvkmftqo.net/BJ5.1?JSESSIONID=20AGRCD3c59DjD_gtLPM16if7ZVmv1_KhPgnNSVD4XGrjy1X4SZ5!98817579-aa178660"
        
        try:
            async with websockets.connect(uri) as ws:
                print("[WebSocket] Connected.")
                await ws.send("iq0")
                
                async def ping_task():
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
                
        except Exception as e:
            print(f"[WebSocket] Error: {e}")
    
    # Run websocket
    try:
        asyncio.run(websocket_worker())
    except KeyboardInterrupt:
        print("[Main] Exiting...")
        buffer_reader.close()
        
if __name__ == "__main__":
    main()