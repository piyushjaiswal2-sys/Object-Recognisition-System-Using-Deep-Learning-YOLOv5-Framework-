import torch

def configure_yolov5m_training():
    """
    Learning: Freezing backbone layers to preserve COCO weights 
    while fine-tuning the 'head' for MOT17 pedestrians.
    """
    model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
    
    # Freeze the first 10 layers (Backbone)
    # This prevents the model from forgetting general features (edges/shapes)
    freeze = [f'model.{x}.' for x in range(10)]
    for k, v in model.named_parameters():
        v.requires_grad = True
        if any(x in k for x in freeze):
            v.requires_grad = False
            
    return model

# data.yaml contents (to be stored in the same folder)
yaml_content = """
train: ../data/images/train
val: ../data/images/val
nc: 1
names: ['pedestrian']
"""