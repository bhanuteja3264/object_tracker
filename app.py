import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import torch
import torchvision.models as models
from torchvision import transforms
import argparse

class ReIdentification:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.resnet18(weights='DEFAULT')
        self.model.fc = torch.nn.Identity()
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image):
        """Extract features from image patch"""
        try:
            if image.size == 0:
                return np.zeros(512)
                
            if len(image.shape) == 2:  # Grayscale to RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA to RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                
            image = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                features = self.model(image)
            return features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return np.zeros(512)
    
    @staticmethod
    def cosine_similarity(a, b):
        """Compute cosine similarity without scikit-learn"""
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0
        return np.dot(a, b) / (a_norm * b_norm)

class PersistentObjectTracker:
    def __init__(self):
        self.yolo = YOLO("yolov8s.pt")  # Will auto-download if not found
        self.reid = ReIdentification()
        self.current_tracks = {}  # Active tracks
        self.archived_tracks = {}  # Disappeared tracks (for re-identification)
        self.track_history = defaultdict(list)
        self.selected_id = None
        self.colors = {}
        self.next_id = 1
        self.max_archive_time = 10.0  # Remember objects for 10 seconds (increased from 5)
        self.target_features = None  # Features of the target object
        self.target_class = None     # Class of the target object
        self.target_history = []     # History of target positions
        self.target_lost_time = None # When target was last seen
        self.is_target_tracking = False # Whether we're actively tracking a target
        
        # Mouse callback
        self.click_point = None
        self.detection_data = None
        
        # Better tracking parameters
        self.reid_threshold = 0.55  # Threshold for re-identification (lowered for better reID)
        self.iou_weight = 0.5       # Weight for IoU in similarity calculation
        self.feature_weight = 0.5   # Weight for feature similarity

    def get_color(self, track_id):
        """Get consistent color for each track ID"""
        if track_id not in self.colors:
            self.colors[track_id] = (
                int((track_id * 50) % 255),
                int((track_id * 100) % 255),
                int((track_id * 150) % 255)
            )
        return self.colors[track_id]

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to select objects"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_point = (x, y)
            if self.detection_data is not None:
                boxes, ids, classes = self.detection_data
                for box, track_id, cls in zip(boxes, ids, classes):
                    x1, y1, x2, y2 = map(int, box)
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        self.selected_id = track_id
                        track = self.current_tracks.get(track_id)
                        if track:
                            # Set this as the target object
                            self.target_features = track['features'].copy()
                            self.target_class = track['class']
                            self.is_target_tracking = True
                            self.target_history = []
                            self.target_lost_time = None
                            print(f"Target selected: {self.yolo.names[int(cls)]} (ID: {track_id})")
                        break

    def update_tracks(self, detections, frame):
        """Main tracking logic with re-identification"""
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        
        # Process current detections
        active_ids = set()
        if detections and len(detections[0].boxes) > 0:
            boxes = detections[0].boxes.xyxy.cpu().numpy()
            classes = detections[0].boxes.cls.cpu().numpy()
            confs = detections[0].boxes.conf.cpu().numpy()
            
            # Try to match with current tracks first
            matched_indices = set()
            for track_id, track in list(self.current_tracks.items()):
                best_match_idx, best_similarity = None, -1
                
                for i, (box, cls) in enumerate(zip(boxes, classes)):
                    if i in matched_indices:
                        continue
                    
                    # Only match same class objects
                    if cls != track['class']:
                        continue
                    
                    # Position similarity (IoU)
                    iou = self.calculate_iou(box, track['last_box'])
                    
                    # Appearance similarity
                    crop = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    if crop.size == 0:
                        continue
                        
                    features = self.reid.extract_features(crop)
                    feature_sim = self.reid.cosine_similarity(features, track['features'])
                    
                    # Combined similarity - give more weight to appearance for better occlusion handling
                    similarity = self.iou_weight * iou + self.feature_weight * feature_sim
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match_idx = i
                
                if best_match_idx is not None and best_similarity > 0.3:  # Lowered threshold for better tracking
                    i = best_match_idx
                    self.current_tracks[track_id].update({
                        'last_box': boxes[i],
                        'features': self.reid.extract_features(
                            frame[int(boxes[i][1]):int(boxes[i][3]), 
                            int(boxes[i][0]):int(boxes[i][2])]),
                        'last_seen': current_time,
                        'class': classes[i],
                        'confidence': confs[i],
                        'reidentified': False
                    })
                    active_ids.add(track_id)
                    matched_indices.add(i)
                    
                    # If this is our target, update target history
                    if track_id == self.selected_id and self.is_target_tracking:
                        box = boxes[i]
                        center = ((int(box[0]) + int(box[2])) // 2, (int(box[1]) + int(box[3])) // 2)
                        self.target_history.append(center)
                        if len(self.target_history) > 60:  # Keep longer history for target
                            self.target_history.pop(0)
                        self.target_lost_time = None
            
            # Handle lost target - prioritize finding it in unmatched detections
            if self.is_target_tracking and self.selected_id not in active_ids:
                best_match_idx, best_target_sim = None, -1
                
                for i, (box, cls) in enumerate(zip(boxes, classes)):
                    if i in matched_indices:
                        continue
                    
                    # Only try to match with same class as target
                    if cls != self.target_class:
                        continue
                    
                    crop = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    if crop.size == 0:
                        continue
                        
                    features = self.reid.extract_features(crop)
                    feature_sim = self.reid.cosine_similarity(features, self.target_features)
                    
                    if feature_sim > best_target_sim and feature_sim > self.reid_threshold:
                        best_target_sim = feature_sim
                        best_match_idx = i
                
                if best_match_idx is not None:
                    # Re-identified our target!
                    i = best_match_idx
                    self.current_tracks[self.selected_id] = {
                        'last_box': boxes[i],
                        'features': self.reid.extract_features(
                            frame[int(boxes[i][1]):int(boxes[i][3]), int(boxes[i][0]):int(boxes[i][2])]),
                        'last_seen': current_time,
                        'class': classes[i],
                        'confidence': confs[i],
                        'reidentified': True
                    }
                    active_ids.add(self.selected_id)
                    matched_indices.add(i)
                    print(f"Target reacquired! ID: {self.selected_id}")
                    
                    # Update target history
                    box = boxes[i]
                    center = ((int(box[0]) + int(box[2])) // 2, (int(box[1]) + int(box[3])) // 2)
                    self.target_history.append(center)
                    if len(self.target_history) > 60:
                        self.target_history.pop(0)
                    self.target_lost_time = None
                    
                    # If we had archived our target, remove it from archive
                    if self.selected_id in self.archived_tracks:
                        del self.archived_tracks[self.selected_id]
            
            # Create new tracks for remaining unmatched detections
            for i, (box, cls) in enumerate(zip(boxes, classes)):
                if i in matched_indices:
                    continue
                
                # Check if this matches any archived track
                best_archived_id, best_archived_sim = None, -1
                for archived_id, archived_track in list(self.archived_tracks.items()):
                    # Skip if archived too long ago
                    if current_time - archived_track['last_seen'] > self.max_archive_time:
                        continue
                    
                    # Only re-identify same class objects
                    if cls != archived_track['class']:
                        continue
                    
                    crop = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    if crop.size == 0:
                        continue
                        
                    features = self.reid.extract_features(crop)
                    feature_sim = self.reid.cosine_similarity(features, archived_track['features'])
                    
                    if feature_sim > best_archived_sim and feature_sim > self.reid_threshold:
                        best_archived_sim = feature_sim
                        best_archived_id = archived_id
                
                if best_archived_id is not None:
                    # Re-identified an archived track
                    track_id = best_archived_id
                    self.current_tracks[track_id] = {
                        'last_box': box,
                        'features': self.reid.extract_features(
                            frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]),
                        'last_seen': current_time,
                        'class': cls,
                        'confidence': confs[i],
                        'reidentified': True
                    }
                    del self.archived_tracks[best_archived_id]
                    print(f"Re-identified object {track_id} after disappearance")
                    
                    # If this is our target, update target info
                    if track_id == self.selected_id and self.is_target_tracking:
                        center = ((int(box[0]) + int(box[2])) // 2, (int(box[1]) + int(box[3])) // 2)
                        self.target_history.append(center)
                        if len(self.target_history) > 60:
                            self.target_history.pop(0)
                        self.target_lost_time = None
                else:
                    # New track
                    track_id = self.next_id
                    self.next_id += 1
                    self.current_tracks[track_id] = {
                        'last_box': box,
                        'features': self.reid.extract_features(
                            frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]),
                        'last_seen': current_time,
                        'class': cls,
                        'confidence': confs[i],
                        'reidentified': False
                    }
                active_ids.add(track_id)
        
        # Archive disappeared tracks
        for track_id in list(self.current_tracks.keys()):
            if track_id not in active_ids:
                self.archived_tracks[track_id] = self.current_tracks[track_id]
                del self.current_tracks[track_id]
                
                # If our target disappeared, note the time
                if track_id == self.selected_id and self.is_target_tracking:
                    self.target_lost_time = current_time
        
        return list(self.current_tracks.items())

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1, y1, x2, y2 = box1
        x1_, y1_, x2_, y2_ = box2
        
        xi1 = max(x1, x1_)
        yi1 = max(y1, y1_)
        xi2 = min(x2, x2_)
        yi2 = min(y2, y2_)
        
        inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_ - x1_) * (y2_ - y1_)
        
        return inter_area / (box1_area + box2_area - inter_area + 1e-6)

    def run(self, source=0):
        cap = cv2.VideoCapture(source)
        cv2.namedWindow("Persistent Object Tracker")
        cv2.setMouseCallback("Persistent Object Tracker", self.mouse_callback)
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            # Run YOLOv8 detection
            detections = self.yolo(frame, verbose=False)
            
            # Update tracks with re-identification
            active_tracks = self.update_tracks(detections, frame)
            
            # Prepare detection data for mouse callback
            if detections and len(detections[0].boxes) > 0:
                self.detection_data = (
                    detections[0].boxes.xyxy.cpu().numpy(),
                    [tid for tid, _ in active_tracks],
                    detections[0].boxes.cls.cpu().numpy()
                )
            else:
                self.detection_data = None
            
            # Visualize results
            annotated_frame = detections[0].plot() if detections and len(detections[0].boxes) > 0 else frame.copy()
            
            # Draw tracking info
            for track_id, track in active_tracks:
                box = track['last_box']
                x1, y1, x2, y2 = map(int, box)
                color = self.get_color(track_id)
                
                # Special color for target object
                if track_id == self.selected_id and self.is_target_tracking:
                    color = (0, 0, 255)  # Red for target
                    
                # Draw thicker box for selected object
                thickness = 3 if track_id == self.selected_id else 2
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Draw tracking path
                if track_id in self.track_history:
                    center = ((x1+x2)//2, (y1+y2)//2)
                    self.track_history[track_id].append(center)
                    if len(self.track_history[track_id]) > 30:
                        self.track_history[track_id].pop(0)
                    
                    points = np.array(self.track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=color, thickness=2)
                else:
                    self.track_history[track_id] = [((x1+x2)//2, (y1+y2)//2)]
                
                # Label with class and ID
                label = f"{self.yolo.names[int(track['class'])]} ID:{track_id}"
                if track.get('reidentified', False):
                    label += " (reID)"
                    
                # Special label for target
                if track_id == self.selected_id and self.is_target_tracking:
                    label = f"TARGET: {label}"
                    
                cv2.putText(annotated_frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Show click point if recently clicked
            if self.click_point:
                cv2.circle(annotated_frame, self.click_point, 5, (0, 0, 255), -1)
                self.click_point = None
            
            # Draw target path for better visualization
            if self.is_target_tracking and len(self.target_history) > 1:
                points = np.array(self.target_history, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 0, 255), thickness=3)
            
            # Display waiting for target reappearance message
            if self.is_target_tracking and self.target_lost_time is not None:
                current_time = cv2.getTickCount() / cv2.getTickFrequency()
                lost_duration = current_time - self.target_lost_time
                if lost_duration <= self.max_archive_time:
                    cv2.putText(annotated_frame, f"Waiting for target reappearance: {int(self.max_archive_time - lost_duration)}s", 
                              (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(annotated_frame, "Target lost permanently", 
                              (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display info
            cv2.putText(annotated_frame, f"Tracking {len(active_tracks)} objects", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(annotated_frame, "Click on object to select target", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Persistent Object Tracker", annotated_frame)
            
            if cv2.waitKey(1) == 27:  # ESC to exit
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='Source video file or camera index')
    args = parser.parse_args()
    
    tracker = PersistentObjectTracker()
    source = int(args.source) if args.source.isdigit() else args.source
    tracker.run(source=source)