import numpy as np
from collections import deque
import time

class PersistentTracker:
    def __init__(self, reid_model=None, max_age=30, min_hits=3):
        self.reid = reid_model
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks = {}
        self.next_id = 0
        
    def init_track(self, bbox, features=None):
        track_id = self.next_id
        self.tracks[track_id] = {
            'bbox': bbox,
            'features': features,
            'age': 0,
            'hits': 1,
            'last_seen': time.time(),
            'history': deque(maxlen=30)
        }
        self.next_id += 1
        return track_id
    
    def update(self, detections, frame):
        updated_objects = []
        
        # Age all tracks
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]['age'] += 1
            
        # Process detections
        for det in detections:
            if len(det) < 6:  # Skip invalid detections
                continue
                
            x1, y1, x2, y2, conf, cls = det[:6]
            bbox = [int(x1), int(y1), int(x2-x1), int(y2-y1)]
            
            # Skip too small detections
            if bbox[2] < 10 or bbox[3] < 10:
                continue
                
            best_match, max_similarity = None, -1
            
            for track_id, track in self.tracks.items():
                # Combined similarity metric (IoU + feature similarity)
                iou = self._calculate_iou(bbox, track['bbox'])
                similarity = iou
                
                if self.reid and track['features'] is not None:
                    try:
                        crop = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                        if crop.size > 0:
                            features = self.reid.extract_features(crop)
                            feature_sim = self._cosine_similarity(features, track['features'])
                            similarity = 0.7*iou + 0.3*feature_sim
                    except:
                        pass
                
                if similarity > 0.3 and similarity > max_similarity:
                    max_similarity = similarity
                    best_match = track_id
            
            if best_match is not None:
                # Update matched track
                self.tracks[best_match]['bbox'] = bbox
                self.tracks[best_match]['age'] = 0
                self.tracks[best_match]['hits'] += 1
                self.tracks[best_match]['last_seen'] = time.time()
                if self.reid:
                    try:
                        crop = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                        if crop.size > 0:
                            self.tracks[best_match]['features'] = self.reid.extract_features(crop)
                    except:
                        pass
                updated_objects.append({'id': best_match, 'bbox': bbox})
            else:
                # New object
                features = None
                if self.reid:
                    try:
                        crop = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                        if crop.size > 0:
                            features = self.reid.extract_features(crop)
                    except:
                        pass
                new_id = self.init_track(bbox, features)
                updated_objects.append({'id': new_id, 'bbox': bbox})
                
        # Remove stale tracks
        current_time = time.time()
        self.tracks = {
            k: v for k, v in self.tracks.items()
            if (v['age'] < self.max_age or v['hits'] >= self.min_hits) 
            and (current_time - v['last_seen']) < 5.0  # Max 5 seconds unseen
        }
        
        return updated_objects
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1+w1, x2+w2)
        yi2 = min(y1+h1, y2+h2)
        
        inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
        box1_area = w1 * h1
        box2_area = w2 * h2
        
        return inter_area / (box1_area + box2_area - inter_area + 1e-6)
    
    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)