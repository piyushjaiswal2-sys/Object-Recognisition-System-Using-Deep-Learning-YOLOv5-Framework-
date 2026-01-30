Markdown
# Object Recognition and Pedestrian Anomaly Detection
### WiDS Course Project 

This repository contains a complete computer vision pipeline developed over four weeks. The system leverages **YOLOv5** for real-time pedestrian detection and **DeepSORT** for multi-object tracking, specifically designed to identify abnormal movement patterns in public spaces.



##  Repository Structure
The project is organized into five modular folders to represent the different stages of the machine learning pipeline:

* **`data_setup/`**: Contains scripts to convert the MOT17 dataset into YOLO-compatible normalized tensors.
* **`models/`**: Includes the training configurations and layer-freezing strategies used for the YOLOv5 detector.
* **`weights/`**: Stores the `best.pt` model weights (the learned parameters for detection).
* **`tracking/`**: Contains the Week 4 implementation of the DeepSORT tracker and trajectory-based anomaly logic.
* **`notebooks/`**: Provides a comprehensive Jupyter/Colab walkthrough for testing the final pipeline.

##  Project Roadmap & Learnings

### Phase 1: CNN Theory & Object Detection (Week 1 & 2)
We started by exploring the mathematical foundations of **Convolutional Neural Networks**, focusing on spatial feature extraction. I implemented the **YOLOv5** architecture, which allows for single-pass inference, making it ideal for real-time surveillance applications.

### Phase 2: Custom Training on MOT17 (Week 3)
I trained a custom detector using the **Multiple Object Tracking 2017 (MOT17)** dataset. This involved:
* Converting raw CSV annotations into normalized center-based coordinates.
* Fine-tuning the YOLOv5m model by freezing backbone layers to utilize transfer learning from the COCO dataset.

### Phase 3: Tracking & Anomaly Detection (Week 4)
Using the **DeepSORT** algorithm, I integrated temporal tracking to assign unique IDs to pedestrians. The system includes an anomaly engine that calculates **velocity vectors** and **displacement** to flag irregular behaviors, such as running or entering forbidden zones.



##  How to Run
1. **Clone the Repo:**
   ```bash
   git clone https://github.com/piyushjaiswal2-sys/Object-Recognisition-System-Using-Deep-Learning-YOLOv5-Framework
1. **Install dependencies**
  ```bash
pip install -r requirements.txt
```
Run the Demo: Open notebooks/final_report-demo.py to see the results on the Avenue dataset.
