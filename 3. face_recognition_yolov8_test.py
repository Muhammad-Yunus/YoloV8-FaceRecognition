# Import Library
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

print("OpenCV version ", cv2.__version__)
print("SuperVision version ", sv.__version__)


# load Yolo V8 ONNX using Ultralytics Yolo
model = "model/yolov8s.onnx"
model = YOLO(model, task='detect')


# create Supervision BoxAnnotator() & label_annotator()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()


# load webcam
cap = cv2.VideoCapture(0)
           
# iterate for each frame in video
while cap.isOpened():
    
    # get image on each frame
    ret, frame = cap.read()
    if not ret:
        break

    # do forward pass (inferencing) yolo v8 onnx
    results = model(frame, imgsz=320)[0]

    # do postprocess detection result
    detections = sv.Detections.from_ultralytics(results)

    # filter detections by confidence level
    confidence_threshold = 0.7 # only show detection box with confidence level > 70%
    valid_indices = detections.confidence >= confidence_threshold
    detections = detections[valid_indices]
    
    # draw bounding box & label 
    box_labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(detections['class_name'], detections.confidence)
    ]
    frame = box_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(scene=frame, detections=detections, labels=box_labels)

    # show result
    cv2.imshow('Frame',frame)

    # wait 1ms per frame and close using 'q'
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()