import torch
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision.transforms as T
from scipy.spatial.distance import cosine
import torchvision.models as models

class CardReIDModel:
    """Card Re-Identification Model that maintains embeddings for consistent card tracking"""
    def __init__(self):
        # Initialize a lightweight model for feature extraction
        self.model = models.mobilenet_v2(weights='DEFAULT')
        self.model.classifier = torch.nn.Identity()  # Remove classification layer for feature extraction
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # GPU optimization: Use half precision for faster computation
        if self.device == 'cuda':
            self.model = self.model.half()
            print("[ReID] Using half precision for faster inference")
        
        # Preprocessing transforms
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Perform a warmup inference to initialize the CUDA context
        if self.device == 'cuda':
            dummy_input = torch.randn(1, 3, 224, 224, device=self.device, dtype=torch.float16)
            with torch.no_grad():
                _ = self.model(dummy_input)
            torch.cuda.synchronize()  # Ensure warm-up is complete
        
        self.card_embeddings = {}  # track_id -> embedding
        self.similarity_threshold = 0.7  # Threshold for considering embeddings similar

    def extract_features(self, frame, boxes):
        """Extract features from card crops"""
        features = []
        
        if frame is None:
            print("Warning: frame is None in extract_features")
            return np.array([np.zeros((1280,))])
            
        if not isinstance(boxes, list) or len(boxes) == 0:
            print("Warning: empty or invalid boxes in extract_features")
            return np.array([np.zeros((1280,))])
        
        # Process boxes in batch when possible for better GPU utilization
        valid_crops = []
        valid_indices = []
        
        with torch.no_grad():
            # Prepare all valid crops for batch processing
            for i, box in enumerate(boxes):
                # Check if box is valid
                if box is None or len(box) < 4:
                    features.append(np.zeros((1280,)))
                    continue
                    
                try:
                    x1, y1, w, h = box
                    x2, y2 = x1 + w, y1 + h
                    
                    # Ensure coordinates are within frame boundaries
                    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
                    
                    if x2 - x1 <= 0 or y2 - y1 <= 0:
                        features.append(np.zeros((1280,)))  # Empty feature vector for invalid boxes
                        continue
                    
                    # Crop card image
                    card_img = frame[y1:y2, x1:x2]
                    if card_img.size == 0:
                        features.append(np.zeros((1280,)))
                        continue
                    
                    # Preprocess image
                    img_tensor = self.transform(card_img)
                    valid_crops.append(img_tensor)
                    valid_indices.append(i)
                    
                except Exception as e:
                    print(f"Feature extraction error: {e}")
                    features.append(np.zeros((1280,)))
                
            # Process all valid crops in one batch
            if valid_crops:
                batch = torch.stack(valid_crops).to(self.device)
                
                # Use half precision for faster inference if on CUDA
                if self.device == 'cuda':
                    batch = batch.half()
                    
                # Extract features in one batch operation
                batch_features = self.model(batch).cpu().numpy()
                
                # Add results back to features list in correct order
                for i, idx in enumerate(valid_indices):
                    # Extend features list to ensure it has enough elements
                    while len(features) <= idx:
                        features.append(np.zeros((1280,)))
                    features[idx] = batch_features[i].flatten()
                    
        if len(features) == 0:
            # Return at least one feature vector of zeros
            return np.array([np.zeros((1280,))])
            
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
            similarity = 1 - cosine(embedding.flatten(), stored_embedding.flatten())  # Higher = more similar
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


class Tracker:
    def __init__(self):
        # 1. Initialize DeepSORT tracker
        self.tracker = DeepSort(
            max_age=30,          # Maximum frames to keep track without detection
            n_init=3,            # Frames needed to confirm a track
            embedder="mobilenet", # Feature extractor for appearance
            max_cosine_distance=0.5,  # Threshold for appearance matching
            max_iou_distance=0.7,     # Threshold for IoU matching
            half=True,          # Use half precision for faster inference
            embedder_gpu=True,   # Use GPU for embedder
            embedder_model_name="mobilenet_v2_batch_norm"  # More efficient model
        )
        
        # 2. Initialize Re-ID model
        self.reid_model = CardReIDModel()
        
        # 3. Initialize Mask R-CNN for occlusion handling
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load a more efficient model for mask detection and use TorchScript for optimization
        if self.device == 'cuda':
            # We'll use a lighter FasterRCNN instead of MaskRCNN for better speed
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            self.mask_rcnn = fasterrcnn_resnet50_fpn(pretrained=True, box_score_thresh=0.7)
            self.mask_rcnn.to(self.device).eval().half()  # Use half precision for speed
            
            # Use torch.jit to optimize if possible
            try:
                # Create a sample input for tracing
                sample_input = [torch.rand(3, 384, 640, device=self.device).half()]
                self.mask_rcnn = torch.jit.trace_module(
                    self.mask_rcnn,
                    {'forward': sample_input},
                    check_trace=False
                )
                print("[Tracker] Using TorchScript optimized object detector")
            except Exception as e:
                print(f"[Tracker] TorchScript optimization failed: {e}, using standard model")
        else:
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            self.mask_rcnn = fasterrcnn_resnet50_fpn(pretrained=True)
            self.mask_rcnn.to(self.device).eval()
            
        self.transform = T.Compose([T.ToTensor()])
        
        # 4. State tracking
        self.active_tracks = {}  # Store active tracks with timestamps
        self.confidence_history = {}  # EMA for confidence scores
        self.detection_threshold = 0.4  # Lower threshold to detect more cards
        
        # 5. Debug detection classes
        self.detected_labels = set()  # Track all unique labels detected
        
        # 6. Optimize tracker parameters
        self.frame_count = 0
        # self.skip_mask_rcnn_frames = 0  # Only run mask detection every N frames
        
        # Perform a warmup inference to initialize the CUDA context
        if self.device == 'cuda':
            torch.cuda.synchronize()  # Wait for GPU to be ready
            dummy_img = torch.zeros(1, 3, 720, 1280, dtype=torch.float16, device=self.device)
            with torch.no_grad():
                self.mask_rcnn([dummy_img])
            torch.cuda.synchronize()  # Ensure model is loaded

    def apply_mask_rcnn(self, frame):
        """Apply object detection to find potential cards"""
        # Skip detection on some frames for better performance
        # self.frame_count += 1
        # if self.frame_count % self.skip_mask_rcnn_frames != 0:
        #     return [], []  # Return two empty lists to match expected unpacking
            
        # Use a smaller input resolution for faster processing
        frame_resized = cv2.resize(frame, (640, 384))
        
        with torch.no_grad():
            # Convert frame to tensor
            img_tensor = self.transform(frame_resized).to(self.device)
            
            # Use half precision on CUDA
            if self.device == 'cuda':
                img_tensor = img_tensor.half()
                
            # Run inference
            predictions = self.mask_rcnn([img_tensor])[0]
            
        # Filter predictions with high confidence
        boxes = []
        scores = []
        
        for i, box in enumerate(predictions['boxes']):
            score = predictions['scores'][i].item()
            if score > 0.7:  # Only keep high confidence detections
                # Scale box coords back to original frame size
                x1, y1, x2, y2 = box.cpu().numpy()
                h_ratio = frame.shape[0] / 384
                w_ratio = frame.shape[1] / 640
                
                x1 *= w_ratio
                x2 *= w_ratio
                y1 *= h_ratio
                y2 *= h_ratio
                
                boxes.append([x1, y1, x2, y2])
                scores.append(score)
                
        return boxes, scores  # Return exactly two values

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

    def update(self, detections, frame):
        """Update tracker with new detections"""
        if frame is None or len(frame.shape) != 3:
            return []
        
        # For larger frames, resize for faster processing
        h, w = frame.shape[:2]
        if w > 1280:  # If very large frame
            scale = 1280 / w
            frame_proc = cv2.resize(frame, (1280, int(h * scale)))
        else:
            frame_proc = frame
            
        # 1. Format yolo detections for DeepSORT
        deepsort_dets = []
        labels = []
        
        for det in detections:
            x1, y1, x2, y2, conf, label = det
            
            # Store label for debugging
            self.detected_labels.add(label)
            
            # Skip low confidence detections
            if conf < self.detection_threshold:
                continue
                
            # Convert to [x1,y1,width,height] format needed by DeepSORT
            w, h = x2 - x1, y2 - y1
            deepsort_dets.append(([x1, y1, w, h], conf, label))
            labels.append(label)

        # Log unique labels periodically
        if len(self.detected_labels) > 0 and len(self.detected_labels) % 10 == 0:
            print(f"[Tracker] Detected label types: {sorted(list(self.detected_labels))}")
            
        try:
            # 2. Apply object detection for additional objects
            object_boxes, object_scores = self.apply_mask_rcnn(frame)
        except Exception as e:
            print(f"[Object Detection Error] {e}")
            object_boxes, object_scores = [], []
        
        # 3. Update DeepSORT tracker
        tracks = self.tracker.update_tracks(deepsort_dets, frame=frame)
        tracked_objects = []
        current_ids = set()
        
        # 4. Process tracked objects
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
            
            # Extract features for Re-ID using our custom ReID model
            features = self.reid_model.extract_features(frame, [[x1, y1, w, h]])
            if len(features) > 0:
                self.reid_model.update_embedding(track_id, features[0])
            
            # Smooth confidence scores
            if hasattr(track, 'det_conf'):
                conf = self.smooth_confidence(track_id, track.det_conf)
            else:
                conf = 0.5  # Default confidence if not available
            
            # Update active tracks
            self.active_tracks[track_id] = {
                'bbox': [x1, y1, x2, y2],
                'label': det_class,
                'confidence': conf,
                'last_seen': time.time()
            }
            
            # Add to result
            tracked_objects.append([x1, y1, x2, y2, track_id, det_class])
            
        # 5. Clean up old tracks
        self.reid_model.clean_old_embeddings(self.active_tracks)
            
        # 6. Check additional object detections for missed cards
        for i, box in enumerate(object_boxes):
            if i < len(object_scores) and object_scores[i] > 0.7:
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                
                # Check if this might be a card that DeepSORT missed
                is_new_detection = True
                for obj in tracked_objects:
                    obj_x1, obj_y1, obj_x2, obj_y2 = obj[0:4]
                    # Check IoU to avoid duplicate detections
                    iou = self.calculate_iou([x1, y1, x2, y2], [obj_x1, obj_y1, obj_x2, obj_y2])
                    if iou > 0.3:
                        is_new_detection = False
                        break
                        
                if is_new_detection:
                    # Extract features for this potential card
                    features = self.reid_model.extract_features(frame, [[x1, y1, w, h]])
                    if len(features) > 0:
                        # Check if it matches any known card by appearance
                        matching_id = self.reid_model.find_matching_card(features[0])
                        
                        if matching_id and matching_id in self.active_tracks:
                            # This is likely a card we've seen before
                            tracked_objects.append([x1, y1, x2, y2, matching_id, self.active_tracks[matching_id]['label']])
                            self.active_tracks[matching_id]['bbox'] = [x1, y1, x2, y2]
                            self.active_tracks[matching_id]['last_seen'] = time.time()
        
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
