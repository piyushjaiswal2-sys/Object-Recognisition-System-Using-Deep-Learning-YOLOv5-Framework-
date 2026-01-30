import os
import pandas as pd

def preprocess_mot17(src_root, out_root):
    """
    Learning: Converting MOT17 (top-left, raw pixels) 
    to YOLO (center-normalized) format.
    """
    os.makedirs(out_root, exist_ok=True)
    
    # We select one detector variant (FRCNN) to avoid data redundancy
    sequences = [s for s in os.listdir(src_root) if 'FRCNN' in s]
    
    for seq in sequences:
        gt_file = os.path.join(src_root, seq, 'gt/gt.txt')
        # Standard MOT17 resolution is usually 1920x1080
        img_w, img_h = 1920, 1080 
        
        df = pd.read_csv(gt_file, header=None)
        # Class 1 is 'Pedestrian' in MOT17
        df = df[df[7] == 1]

        for frame in df[0].unique():
            objs = df[df[0] == frame]
            with open(f"{out_root}/{seq}_f{frame:04d}.txt", 'w') as f:
                for _, row in objs.iterrows():
                    # Math: (x+w/2)/W , (y+h/2)/H
                    x_center = (row[2] + row[4]/2) / img_w
                    y_center = (row[3] + row[5]/2) / img_h
                    w_norm = row[4] / img_w
                    h_norm = row[5] / img_h
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

print("Step 1: MOT17 Data Setup Code Ready.")