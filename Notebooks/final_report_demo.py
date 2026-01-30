from models.train_config import configure_yolov5m_training
from tracking.tracker_engine import AnomalyEngine
import cv2

# Initialize System
# Note: In a real run, you'd use 'weights/best.pt'
engine = AnomalyEngine(max_dist=25)
cap = cv2.VideoCapture('test_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # Process through the pipeline
    # 1. YOLO Detection (Week 2/3)
    # 2. DeepSORT Tracking (Week 4)
    # 3. Anomaly Flagging (Week 4)
    processed_frame = engine.update(frame, []) # detections would go here
    
    cv2.imshow('Final Project Output', processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()