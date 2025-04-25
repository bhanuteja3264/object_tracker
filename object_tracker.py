import cv2
import numpy as np
import configparser
from utils.helpers import draw_box, resize_frame

class ObjectTracker:
    def __init__(self, config_path="configs/tracker_config.ini"):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        
        # Initialize detector
        self.net = cv2.dnn.readNetFromCaffe(
            self.config["detector"]["config_path"],
            self.config["detector"]["model_path"]
        )
        
        # Initialize tracker
        self.tracker = None
        self.tracking = False
        self.selected_object = None
        self.CLASSES = ["background", "person", "bicycle", "car", "motorcycle",
                        "airplane", "bus", "train", "truck", "boat", "traffic light",
                        "fire hydrant", "stop sign", "parking meter", "bench", "bird",
                        "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
                        "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
                        "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                        "kite", "baseball bat", "baseball glove", "skateboard",
                        "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
                        "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
                        "cake", "chair", "couch", "potted plant", "bed", "dining table",
                        "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
                        "cell phone", "microwave", "oven", "toaster", "sink",
                        "refrigerator", "book", "clock", "vase", "scissors",
                        "teddy bear", "hair drier", "toothbrush"]

    def detect_objects(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()
        
        objects = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > float(self.config["detector"]["confidence_threshold"]):
                class_id = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                objects.append({
                    "class_id": class_id,
                    "confidence": confidence,
                    "box": (x1, y1, x2-x1, y2-y1),  # (x,y,w,h)
                    "class_name": self.CLASSES[class_id]
                })
        return objects

    def init_tracker(self, frame, bbox):
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, bbox)
        self.tracking = True

    def update_tracker(self, frame):
        if self.tracker is None:
            return False, None
        return self.tracker.update(frame)

def main():
    tracker = ObjectTracker()
    cap = cv2.VideoCapture(0)
    
    cv2.namedWindow("Object Tracker")
    print("Instructions:\n1. Click on any object to track it\n2. Press 'r' to reset\n3. Press 'q' to quit")

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not tracker.tracking:
            for obj in detected_objects:
                x1, y1, w, h = obj["box"]
                if x1 <= x <= x1+w and y1 <= y <= y1+h:
                    tracker.init_tracker(frame, obj["box"])
                    tracker.selected_object = obj
                    break

    cv2.setMouseCallback("Object Tracker", mouse_callback)
    
    detected_objects = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame = resize_frame(frame, width=int(tracker.config["tracker"]["resize_width"]))
        
        if not tracker.tracking:
            detected_objects = tracker.detect_objects(frame)
            if tracker.config["tracker"].getboolean("show_all_detections"):
                for obj in detected_objects:
                    label = f"{obj['class_name']}: {obj['confidence']:.2f}"
                    draw_box(frame, obj["box"], label, (255, 0, 0))
            
            cv2.putText(frame, "Click on object to track", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            success, bbox = tracker.update_tracker(frame)
            if success:
                label = f"Tracking: {tracker.selected_object['class_name']}"
                draw_box(frame, bbox, label, (0, 255, 0))
            else:
                tracker.tracking = False
                tracker.selected_object = None
        
        cv2.putText(frame, "Press 'r' to reset | 'q' to quit", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Object Tracker", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            tracker.tracking = False
            tracker.selected_object = None
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()