import os
import time
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from PIL import Image
from torchvision import transforms
import torchvision
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

class NeuralVal(Dataset):
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
    transforms.ToTensor(),
])
dataset = NeuralVal(img_dir='MaskImages', 
                            annotations_file='annotations.csv', 
                            transform=transform)
train_loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Model setup
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
num_classes = 2  # 1 class (serial number) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)

# Load the saved model if it exists
model_save_path = "fasterrcnn_model.pth"
if os.path.exists(model_save_path):
    print(f"Loading model from {model_save_path}")
    model.load_state_dict(torch.load(model_save_path))
else:
    print(f"No saved model found at {model_save_path}. Starting training from scratch.")

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
def train(train=True):
    print("Training Started...")
    if train:
        total_iterations = num_epochs * len(train_loader)
        progress = 0
        current_iteration = 0
        start_time = time.time()
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            for images, targets in train_loader:
                iter_start_time = time.time() #Log the start time
                # Training
                images = list(image.to(device) for image in images)  
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                optimizer.zero_grad()
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                optimizer.step()
                epoch_loss += losses.item()
                current_iteration += 1

                # Progress Management
                progress = current_iteration / total_iterations
                iter_end_time = time.time()  # Log the end time
                time_passed = iter_end_time - iter_start_time
                elapsed_time = time.time() - start_time
                avg_time_per_iteration = elapsed_time / current_iteration
                time_left = round(((total_iterations - current_iteration) * avg_time_per_iteration) / 60, 2)
                
                print(f"Iteration {current_iteration}/{total_iterations} ({round(progress*100,2)}%), Current epoch Loss: {epoch_loss}, ETA: {time_left} min")

            print(f'Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader)}')
            
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

train()
