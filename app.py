import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torchvision.models as models
from torchvision import transforms
import time

class ReIdentification:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.resnet18(weights="DEFAULT")
        self.model.fc = torch.nn.Identity()
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, image):
        """Extract features from an image patch."""
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
        """Compute cosine similarity without scikit-learn."""
        if a is None or b is None:
            return 0  # Return 0 if either vector is None
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0
        return np.dot(a, b) / (a_norm * b_norm)

class TargetObjectTracker:
    def __init__(self):
        self.yolo = YOLO("yolov8n.pt")  # Nano model for speed
        self.reid = ReIdentification()

        # Target tracking parameters
        self.target_id = None
        self.target_class = None
        self.target_features = None
        self.target_box = None
        self.all_objects_mode = True

        # Occlusion and out-of-frame tracking
        self.target_visible = True
        self.occlusion_start_time = None
        self.max_occlusion_time = 3.0
        self.reidentification_time = None

        # Mouse selection
        self.click_point = None
        self.detection_data = None

        # Performance metrics
        self.fps = 0
        self.frame_count = 0
        self.start_time = None

        # Reset button area
        self.reset_button = [10, 10, 100, 40]

    def reset_tracking(self):
        """Reset target tracking."""
        self.target_id = None
        self.target_class = None
        self.target_features = None
        self.target_box = None
        self.all_objects_mode = True
        self.target_visible = True
        self.occlusion_start_time = None
        self.reidentification_time = None
        print("Tracking reset - showing all objects")

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for object selection and reset."""
        frame = param  # Access the frame via the param argument
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if reset button was clicked
            if (self.reset_button[0] <= x <= self.reset_button[0] + self.reset_button[2] and
                self.reset_button[1] <= y <= self.reset_button[1] + self.reset_button[3]):
                self.reset_tracking()
                return

            # Select object based on click
            self.click_point = (x, y)
            if self.detection_data is not None:
                boxes, classes, ids = self.detection_data
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        self.target_id = ids[i]
                        self.target_class = classes[i]
                        self.target_box = box
                        self.all_objects_mode = False
                        self.target_visible = True
                        self.target_features = self.reid.extract_features(
                            frame[int(y1):int(y2), int(x1):int(x2)]
                        )
                        print(f"Target selected: {self.yolo.names[int(classes[i])]} (ID: {self.target_id})")
                        break

    def run(self, source=0):
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        cv2.namedWindow("Target Object Tracker")
        cv2.setMouseCallback("Target Object Tracker", self.mouse_callback, param=None)

        self.start_time = time.time()
        self.yolo.conf = 0.5

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Update mouse callback parameter
            cv2.setMouseCallback("Target Object Tracker", self.mouse_callback, param=frame)

            # Calculate FPS
            current_time = time.time()
            self.frame_count += 1
            if current_time - self.start_time >= 1:
                self.fps = self.frame_count / (current_time - self.start_time)
                self.frame_count = 0
                self.start_time = current_time

            detections = self.yolo(frame, verbose=False)
            annotated_frame = frame.copy()

            if self.all_objects_mode:
                if detections and len(detections[0].boxes) > 0:
                    boxes = detections[0].boxes.xyxy.cpu().numpy()
                    classes = detections[0].boxes.cls.cpu().numpy()
                    ids = np.arange(1, len(boxes) + 1)

                    self.detection_data = (boxes, classes, ids)
                    for box, cls, obj_id in zip(boxes, classes, ids):
                        x1, y1, x2, y2 = map(int, box)
                        color = (0, 255, 0)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{self.yolo.names[int(cls)]} ID:{obj_id}"
                        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    cv2.putText(annotated_frame, "Click on an object to track", (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                found_target = False
                if detections and len(detections[0].boxes) > 0:
                    boxes = detections[0].boxes.xyxy.cpu().numpy()
                    classes = detections[0].boxes.cls.cpu().numpy()
                    ids = np.arange(1, len(boxes) + 1)

                    for box, cls, obj_id in zip(boxes, classes, ids):
                        x1, y1, x2, y2 = map(int, box)
                        cropped = frame[y1:y2, x1:x2]
                        features = self.reid.extract_features(cropped)
                        similarity = self.reid.cosine_similarity(self.target_features, features)

                        if similarity > 0.8:  # Re-identification threshold
                            self.target_id = obj_id
                            self.target_box = box
                            self.target_visible = True
                            self.occlusion_start_time = None
                            if self.reidentification_time:
                                elapsed = current_time - self.reidentification_time
                                print(f"Target is found after {elapsed:.1f} seconds.")
                                self.reidentification_time = None
                            color = (0, 0, 255)
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            label = f"TARGET: {self.yolo.names[int(cls)]}"
                            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                            found_target = True
                            break

                if not found_target:
                    if self.target_visible:
                        self.occlusion_start_time = current_time
                        self.target_visible = False
                        self.reidentification_time = current_time
                    elapsed_occlusion_time = current_time - (self.occlusion_start_time or current_time)
                    if elapsed_occlusion_time > self.max_occlusion_time:
                        status = "Target is out of frame."
                    else:
                        status = f"Target is obstructed for {elapsed_occlusion_time:.1f}s."
                    cv2.putText(annotated_frame, status, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Draw reset button
            x, y, w, h = self.reset_button
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (50, 50, 50), -1)
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (200, 200, 200), 2)
            cv2.putText(annotated_frame, "RESET", (x + 15, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            # Display FPS
            cv2.putText(annotated_frame, f"FPS: {self.fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            mode_text = "Mode: All Objects" if self.all_objects_mode else "Mode: Target Only"
            cv2.putText(annotated_frame, mode_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Target Object Tracker", annotated_frame)
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break
            elif key == ord('r'):
                self.reset_tracking()

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = TargetObjectTracker()
    tracker.run(source=0)