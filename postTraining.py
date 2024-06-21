import os
import pandas as pd
from PIL import Image
import torch
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

def load_model(model_load_path):
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    num_classes = 2  # 1 class (serial number) + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_load_path))
    model.eval()
    model.to(device)
    return model

def red_mask_intensity(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read the image {image_path}.")
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lower_red = np.array([100, 0, 0])
    upper_red = np.array([255, 100, 100])
    mask = cv2.inRange(image_rgb, lower_red, upper_red)
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    gray_mask_float = gray_mask.astype(np.float32) / 255.0
    return gray_mask_float

def draw_bounding_boxes(image, predictions, threshold=0.80):
    image = image.cpu().numpy().transpose(1, 2, 0)
    image = (image * 255).astype(np.uint8)
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    for i, (box, score) in enumerate(zip(predictions['boxes'], predictions['scores'])):
        if score.item() >= threshold:
            x_min, y_min, x_max, y_max = box.float().cpu().numpy()
            label = predictions['labels'][i].item()
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            #print(x_min, y_min, x_max, y_max)
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            ax.text(x_min, y_min - 10, f'Score: {score:.2f}', color='green', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    plt.axis('off')
    plt.show()
    
def predict(image_path, model, device):
    image = red_mask_intensity(image_path)
    image_tensor = F.to_tensor(image).to(device)
    
    with torch.no_grad():
        predictions = model([image_tensor])[0]
    return predictions

def display_prediction(original_image,predictions,threshold=0.80):
    image = Image.open(original_image).convert("RGB")
    image_tensor = F.to_tensor(image).to(device)
    draw_bounding_boxes(image_tensor, predictions,threshold)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model_load_path = "fasterrcnn_model2.pth"
model = load_model(model_load_path)
original_image = 'image628.jpg'
# masked_name = 'image17'
predictions = predict(original_image,model,device)
box = predictions['boxes'][0].cpu().numpy()
label = predictions['labels'][0].cpu().numpy()
score = predictions['scores'][0].cpu().numpy()
print("Box Cords",box, "Label", label, "Score", score)
display_prediction(original_image,predictions,threshold=0.7)
# directory = 'Temp'
# for filename in os.listdir(directory):
#     file_path = os.path.join(directory,filename)
#     print(file_path)
#     predictions = predict(file_path,model,device)
#     print(predictions['boxes'][0].cpu().numpy())
