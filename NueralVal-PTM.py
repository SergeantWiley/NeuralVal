import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
import torch.optim as optim
from torchvision.models.detection import fasterrcnn_resnet50_fpn
class NueralVal(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None):
        self.img_dir = img_dir
        self.annotations = pd.read_csv(annotations_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        boxes = self.annotations.iloc[idx, 1:5].values.astype('float').reshape(-1, 4)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        if self.transform:
            image = self.transform(image)
        target = {'boxes': boxes, 'labels': labels}
        return image, target

transform = transforms.Compose([
    #transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = NueralVal(img_dir='MaskImages', 
                            annotations_file='annotations.csv', 
                            transform=transform)

train_loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # 1 class (serial number) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Model setup
model = fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # 1 class (serial number) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Training loop
num_epochs = 10

# for epoch in range(num_epochs):
#     model.train()
#     epoch_loss = 0
#     for images, targets in train_loader:
#         images = list(image.to(device) for image in images)
#         print("Images to tensor")
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#         print("Targets to tensor")
#         optimizer.zero_grad()
#         print("Optimizer set")
#         loss_dict = model(images, targets)
#         print("Losses found")
#         losses = sum(loss for loss in loss_dict.values())
#         print("Losses compiled and calculated")
#         losses.backward()
#         print("Backward Propagation Completed")
#         optimizer.step()
#         print("Going to next step")
#         epoch_loss += losses.item()

#     print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader)}')
#model_save_path = "fasterrcnn_model.pth"
#torch.save(model.state_dict(), model_save_path)


model_load_path = "fasterrcnn_model.pth"
model.load_state_dict(torch.load(model_load_path))
model.eval()  # Set the model to evaluation mode

import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F
import numpy as np
def draw_bounding_boxes(image, predictions, threshold=0.005):
    # Convert the image from a tensor to a NumPy array and transpose the dimensions
    image = image.cpu().numpy().transpose(1, 2, 0)
    image = (image * 255).astype(np.uint8)  # Convert to uint8

    # Print all bounding box details for debugging
    print("All bounding boxes and scores:")
    for i, (box, score) in enumerate(zip(predictions['boxes'], predictions['scores'])):
        x_min, y_min, x_max, y_max = box.float
        ().cpu().numpy()
        print(f'Box {i}: ({x_min}, {y_min}), ({x_max}, {y_max}), Score: {score.item()}')
    
    # Loop over the predictions and draw the bounding boxes
    for i, box in enumerate(predictions['boxes']):
        score = predictions['scores'][i].item()
        if score >= threshold:
            x_min, y_min, x_max, y_max = box.float().cpu().numpy()
            label = predictions['labels'][i].item()
            
            # Draw the bounding box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)
            cv2.putText(image, f'Score: {score:.2f}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return image

def run_inference_and_draw(image_path, model, device):
    # Load the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).to(device)
    
    # Run inference
    with torch.no_grad():
        predictions = model([image_tensor])[0]
    print(f"predictions: {predictions}")
    
    # Draw bounding boxes
    result_image = draw_bounding_boxes(image_tensor, predictions)
    
    # Display the result
    plt.imshow(result_image)
    plt.axis('off')
    plt.show()

# Example usage
image_path = 'image11.jpg'
run_inference_and_draw(image_path, model, device)