import cv2
import numpy as np

def draw_box(frame, box, label, color=(0, 255, 0), thickness=2):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def resize_frame(frame, width=None):
    if width is None:
        return frame
    h, w = frame.shape[:2]
    r = width / float(w)
    dim = (width, int(h * r))
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)