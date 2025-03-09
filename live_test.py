import asyncio
import json
import threading
import io
import time
import os
import cv2
import av
import torch
import websockets
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
 
# Verify numpy version
assert np.__version__.startswith('1.'), f"Invalid numpy version {np.__version__} - must be 1.x"

# Load the trained YOLO model
model = YOLO("./1-3_blackjack.pt")

class CompatTracker:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=40,
            embedder="mobilenet",
            half=False,
            max_cosine_distance=0.4,
            max_iou_distance=0.6,
            nms_max_overlap=1.0
        )
        self.track_history = {}
        self.last_seen_threshold = 5

    def update(self, detections, frame):
        ds_detections = [([x1, y1, x2, y2], conf, label) 
                        for x1, y1, x2, y2, conf, label in detections]

        tracks = self.tracker.update_tracks(ds_detections, frame=frame)
        current_ids = set()
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()
            
            self.track_history[track_id] = {
                'bbox': ltrb,
                'label': track.get_det_class(),
                'last_seen': time.time(),
                'positions': self.track_history.get(track_id, {}).get('positions', []) + [ltrb]
            }
            current_ids.add(track_id)

        # Cleanup old tracks
        for track_id in list(self.track_history.keys()):
            if track_id not in current_ids:
                if time.time() - self.track_history[track_id]['last_seen'] > self.last_seen_threshold:
                    del self.track_history[track_id]

        return self.track_history

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
            return bytes(self.buffer[:n]) if n > 0 else bytes(self.buffer)

    def close(self):
        with self.lock:
            self.closed_flag = True
            self.data_available.notify_all()

buffer_reader = BufferReader()
tracker = CompatTracker()

def decoder_worker():
    window_initialized = False
    try:
        container = av.open(buffer_reader, format='mp4', mode='r')
        frame_skip = 3
        frame_count = 0

        for frame in container.decode(video=0):
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            img = frame.to_ndarray(format='bgr24')
            if img is None:
                continue

            img = cv2.resize(img, (640, 480))
            results = model(img, conf=0.4, imgsz=320)
            detections = []
            
            for result in results:
                for box in result.boxes:
                    x1, y1 = map(int, box.xyxy[0][:2])
                    x2, y2 = map(int, box.xyxy[0][2:])
                    label = result.names[int(box.cls[0])]
                    conf = box.conf[0].item()
                    detections.append((x1, y1, x2, y2, conf, label))

            tracked_objects = tracker.update(detections, img)

            try:
                if not window_initialized:
                    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
                    window_initialized = True
                
                for track_id, info in tracked_objects.items():
                    x1, y1, x2, y2 = map(int, info['bbox'])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f'ID{track_id} {info["label"]}', (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                
                cv2.imshow('Detection', img)
                if cv2.waitKey(1) == ord('q'):
                    break
            except Exception as e:
                print(f"Display error: {e}")

    except Exception as e:
        print(f"Decoder error: {e}")
    finally:
        cv2.destroyAllWindows()

def start_decoder():
    threading.Thread(target=decoder_worker, daemon=True).start()

async def websocket_worker():
    uri = "wss://ws1.zigdpseatvkmftqo.net/BJ5.1?JSESSIONID=20AGRCD3c59DjD_gtLPM16if7ZVmv1_KhPgnNSVD4XGrjy1X4SZ5!98817579-aa178660"
    
    try:
        async with websockets.connect(uri) as ws:
            print("Connected to WebSocket")
            await ws.send("iq0")

            async def ping_task():
                counter = 1
                while True:
                    await asyncio.sleep(30)
                    await ws.send(json.dumps({
                        "cmd": "ping",
                        "counter": counter,
                        "clientTime": int(time.time() * 1000)
                    }))
                    counter += 1

            asyncio.create_task(ping_task())

            async for message in ws:
                if isinstance(message, bytes):
                    buffer_reader.feed(message)

    except Exception as e:
        print(f"WebSocket error: {e}")

if __name__ == "__main__":
    start_decoder()
    try:
        asyncio.run(websocket_worker())
    except KeyboardInterrupt:
        print("Exiting...")
        buffer_reader.close()