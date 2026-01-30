import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

class AnomalyEngine:
    """
    Learning: Spatiotemporal analysis using DeepSORT tracking IDs 
    to calculate velocity vectors and flag anomalies.
    """
    def __init__(self, max_dist=20):
        self.tracker = DeepSort(max_age=30)
        self.trajectories = {} # {id: [(x1,y1), (x2,y2)...]}
        self.anomaly_threshold = max_dist

    def update(self, frame, detections):
        # Update tracker with detections from YOLOv5
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        for track in tracks:
            if not track.is_confirmed(): continue
            
            tid = track.track_id
            bbox = track.to_ltrb() # Left, Top, Right, Bottom
            cx, cy = int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2)
            
            # Store history for trajectory drawing (Week 4 Learning)
            self.trajectories.setdefault(tid, []).append((cx, cy))
            
            # ANOMALY LOGIC: Speed Check
            is_anomaly = False
            if len(self.trajectories[tid]) > 5:
                start_pt = self.trajectories[tid][-5]
                # Euclidean distance math
                velocity = np.sqrt((cx-start_pt[0])**2 + (cy-start_pt[1])**2)
                if velocity > self.anomaly_threshold:
                    is_anomaly = True

            # Visual Feedback
            color = (0, 0, 255) if is_anomaly else (0, 255, 0)
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.putText(frame, f"ID {tid}", (int(bbox[0]), int(bbox[1]-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        return frame