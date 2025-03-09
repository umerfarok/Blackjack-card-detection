import torch
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import torchvision.transforms as T
from scipy.spatial.distance import cosine

class LightweightReIDModel:
    """Lightweight Re-Identification Model for CPU usage"""
    def __init__(self):
        # Use built-in OpenCV HOG descriptor for feature extraction (CPU friendly)
        self.hog = cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
        self.feature_size = 3780  # HOG feature size
        
        # Transform for preprocessing
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 64)),  # HOG standard size
            T.ToTensor(),
        ])
        
        self.card_embeddings = {}  # track_id -> embedding
        self.similarity_threshold = 0.6  # Threshold for considering embeddings similar
        
        print("Initialized lightweight CPU-friendly feature extractor")

    def extract_features(self, frame, boxes):
        """Extract HOG features from card crops"""
        features = []
        
        if frame is None:
            print("Warning: frame is None in extract_features")
            return np.array([np.zeros((self.feature_size,))])
            
        if not isinstance(boxes, list) or len(boxes) == 0:
            print("Warning: empty or invalid boxes in extract_features")
            return np.array([np.zeros((self.feature_size,))])
        
        for box in boxes:
            # Check if box is valid
            if box is None or len(box) < 4:
                features.append(np.zeros((self.feature_size,)))
                continue
                
            try:
                x1, y1, w, h = box
                x2, y2 = x1 + w, y1 + h
                
                # Ensure coordinates are within frame boundaries
                x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
                
                if x2 - x1 <= 0 or y2 - y1 <= 0:
                    features.append(np.zeros((self.feature_size,)))
                    continue
                
                # Crop card image
                card_img = frame[y1:y2, x1:x2]
                if card_img.size == 0:
                    features.append(np.zeros((self.feature_size,)))
                    continue
                
                # Resize to HOG standard size
                card_img = cv2.resize(card_img, (64, 128))
                
                # Extract HOG features
                hog_features = self.hog.compute(card_img)
                features.append(hog_features.flatten())
            except Exception as e:
                print(f"Feature extraction error: {e}")
                features.append(np.zeros((self.feature_size,)))
                
        if len(features) == 0:
            # Return at least one feature vector of zeros
            return np.array([np.zeros((self.feature_size,))])
            
        return np.array(features)

    def update_embedding(self, track_id, embedding):
        """Store or update embedding for a track_id"""
        if track_id not in self.card_embeddings:
            self.card_embeddings[track_id] = embedding
        else:
            # Update using exponential moving average
            alpha = 0.7
            self.card_embeddings[track_id] = alpha * self.card_embeddings[track_id] + (1 - alpha) * embedding

    def find_matching_card(self, embedding):
        """Find existing card with most similar embedding"""
        best_match = None
        best_similarity = 0
        
        for track_id, stored_embedding in self.card_embeddings.items():
            # Handle different sizes if needed
            min_len = min(len(embedding.flatten()), len(stored_embedding.flatten()))
            similarity = 1 - cosine(embedding.flatten()[:min_len], stored_embedding.flatten()[:min_len])
            
            if similarity > self.similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = track_id
                
        return best_match

    def clean_old_embeddings(self, active_tracks, max_age=300):
        """Remove embeddings for tracks that haven't been seen for a while"""
        current_time = time.time()
        for track_id in list(self.card_embeddings.keys()):
            if track_id not in active_tracks:
                if current_time - active_tracks.get(track_id, {}).get('last_seen', 0) > max_age:
                    del self.card_embeddings[track_id]


class CPUTracker:
    def __init__(self):
        # 1. Initialize DeepSORT tracker with lightweight settings
        self.tracker = DeepSort(
            max_age=30,          # Maximum frames to keep track without detection
            n_init=3,            # Frames needed to confirm a track
            embedder=None,       # No embedder for faster CPU performance
            max_cosine_distance=0.7,  # Higher threshold for matching
            max_iou_distance=0.8,     # Higher threshold for IoU
        )
        
        # 2. Initialize lightweight Re-ID model
        self.reid_model = LightweightReIDModel()
        
        # 3. No MaskRCNN for CPU version - too resource intensive
        
        # 4. State tracking
        self.active_tracks = {}  # Store active tracks with timestamps
        self.confidence_history = {}  # EMA for confidence scores
        self.detection_threshold = 0.25  # Minimum confidence for detection
        self.last_frame_tracks = []  # Store last frame's tracks for motion prediction
        
    def smooth_confidence(self, track_id, new_conf):
        """Apply exponential moving average to confidence scores"""
        # Check if new_conf is None
        if new_conf is None:
            if track_id in self.confidence_history:
                # Return existing confidence if we have history
                return self.confidence_history[track_id]
            else:
                # Return default confidence if no history
                return 0.5
                
        alpha = 0.8  # Higher alpha means more weight to historical values
        
        if track_id not in self.confidence_history:
            self.confidence_history[track_id] = new_conf
        else:
            # Make sure confidence_history[track_id] is not None
            if self.confidence_history[track_id] is None:
                self.confidence_history[track_id] = new_conf
            else:
                self.confidence_history[track_id] = alpha * self.confidence_history[track_id] + (1-alpha) * new_conf
                
        return self.confidence_history[track_id]

    def predict_missing_detections(self, tracked_objects):
        """Use simple motion prediction for occluded or missing cards"""
        # This is a simplified version that doesn't use heavyweight models like Mask R-CNN
        
        # Create motion vectors for existing tracks
        predictions = {}
        for track_id, track_info in self.active_tracks.items():
            last_seen = track_info.get('last_seen', time.time())
            time_since_update = time.time() - last_seen
            
            # Only predict for recently lost tracks (within 1 second)
            if time_since_update < 1.0 and time_since_update > 0.1:
                # If we have historical positions, we can predict next position
                positions = track_info.get('positions', [])
                if len(positions) >= 2:
                    x1, y1, x2, y2 = positions[-1]
                    prev_x1, prev_y1, prev_x2, prev_y2 = positions[-2]
                    
                    # Calculate velocity (very basic motion prediction)
                    dx = (x1 - prev_x1) * 0.5  # Scale down motion for smoother prediction
                    dy = (y1 - prev_y1) * 0.5
                    dw = ((x2 - x1) - (prev_x2 - prev_x1)) * 0.5
                    dh = ((y2 - y1) - (prev_y2 - prev_y1)) * 0.5
                    
                    # Predict next position
                    pred_x1 = int(x1 + dx)
                    pred_y1 = int(y1 + dy)
                    pred_x2 = int(x2 + dx + dw)
                    pred_y2 = int(y2 + dy + dh)
                    
                    # Store predicted position
                    predictions[track_id] = [pred_x1, pred_y1, pred_x2, pred_y2]
        
        # Add predictions for tracks that haven't been detected
        for track_id, pred_bbox in predictions.items():
            # Check if this track ID is already in current detections
            found = False
            for obj in tracked_objects:
                if obj[4] == track_id:
                    found = True
                    break
            
            # If not found, add it as a prediction
            if not found:
                label = self.active_tracks[track_id].get('label', 'unknown')
                tracked_objects.append([*pred_bbox, track_id, label])
                # Mark this as a prediction
                self.active_tracks[track_id]['predicted'] = True
                
        return tracked_objects

    def update(self, detections, frame):
        """Update tracker with new detections"""
        if frame is None or len(frame.shape) != 3:
            return []
            
        # 1. Format yolo detections for DeepSORT
        deepsort_dets = []
        labels = []
        
        for det in detections:
            x1, y1, x2, y2, conf, label = det
            
            # Skip low confidence detections
            if conf < self.detection_threshold:
                continue
                
            # Convert to [x1,y1,width,height] format needed by DeepSORT
            w, h = x2 - x1, y2 - y1
            deepsort_dets.append(([x1, y1, w, h], conf, label))
            labels.append(label)
            
        # 2. Update DeepSORT tracker
        tracks = self.tracker.update_tracks(deepsort_dets, frame=frame)
        tracked_objects = []
        current_ids = set()
        
        # 3. Process tracked objects
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            current_ids.add(track_id)
            
            # Get track bounding box in [x1,y1,x2,y2] format
            x1, y1, x2, y2 = track.to_ltrb()  
            w, h = x2 - x1, y2 - y1
            
            # Get detection class (card type)
            det_class = track.get_det_class()
            
            # Extract features for Re-ID using our lightweight ReID model
            features = self.reid_model.extract_features(frame, [[x1, y1, w, h]])
            if len(features) > 0:
                self.reid_model.update_embedding(track_id, features[0])
            
            # Smooth confidence scores
            if hasattr(track, 'det_conf'):
                conf = self.smooth_confidence(track_id, track.det_conf)
            else:
                conf = 0.5  # Default confidence if not available
            
            # Update active tracks
            if track_id in self.active_tracks:
                # Update existing track
                self.active_tracks[track_id].update({
                    'bbox': [x1, y1, x2, y2],
                    'label': det_class,
                    'confidence': conf,
                    'last_seen': time.time(),
                    'predicted': False
                })
                
                # Add position to history
                positions = self.active_tracks[track_id].get('positions', [])
                positions.append([x1, y1, x2, y2])
                # Keep only the last 5 positions
                if len(positions) > 5:
                    positions = positions[-5:]
                self.active_tracks[track_id]['positions'] = positions
            else:
                # New track
                self.active_tracks[track_id] = {
                    'bbox': [x1, y1, x2, y2],
                    'label': det_class,
                    'confidence': conf,
                    'last_seen': time.time(),
                    'positions': [[x1, y1, x2, y2]],
                    'predicted': False
                }
            
            # Add to result
            tracked_objects.append([x1, y1, x2, y2, track_id, det_class])
            
        # 4. Clean up old tracks
        self.reid_model.clean_old_embeddings(self.active_tracks)
        current_time = time.time()
        for track_id in list(self.active_tracks.keys()):
            if current_time - self.active_tracks[track_id]['last_seen'] > 5.0:  # 5 seconds timeout
                del self.active_tracks[track_id]
        
        # 5. Predict missing detections (replaces Mask R-CNN for CPU)
        tracked_objects = self.predict_missing_detections(tracked_objects)
        
        return tracked_objects
        
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes [x1,y1,x2,y2]"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0