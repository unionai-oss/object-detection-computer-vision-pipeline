"""
Object Detection Workflow with Flyte and PyTorch using the Faster R-CNN model

"""

import torch
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import box_iou
from flytekit import task, workflow

# Check and set the available device
device = (
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")

hyperparams = { 
    "batch_size": 2,
    "num_workers": 4,
    "lr": 0.005,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "step_size": 3,
    "gamma": 0.1,
    "num_epochs": 10
}

# --------------------------------
# load data set (create data loader) - task
# --------------------------------
@task
def load_dataset():
    # Define the custom collate function
    def collate_fn(batch):
        return tuple(zip(*batch))

    # Define the transformations for the images
    transform = T.Compose([T.ToTensor()])

    # Load the dataset
    dataset = CocoDetection(root='/content/swag', annFile='/content/swag/train.json', transform=transform)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)
    return data_loader

# --------------------------------
# donwload model - task
# --------------------------------

@task
def download_model():
    # Load a pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # model.to(device)
    return model

# --------------------------------
# train model - task
# --------------------------------
def train_model
# Replace the classifier with a new one (number of classes + 1 for the background)
num_classes = 3  # number of classes + background (TODO: add one for the background class automatically)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

model.to(device)

# Define optimizer and learning rate
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# Function to filter out and correct invalid boxes
def filter_and_correct_boxes(targets):
    filtered_targets = []
    for target in targets:
        boxes = target['boxes']
        labels = target['labels']
        valid_indices = []
        for i, box in enumerate(boxes):
            if box[2] > box[0] and box[3] > box[1]:
                valid_indices.append(i)
            else:
                print(f"Invalid box found and removed: {box}")
        filtered_boxes = boxes[valid_indices]
        filtered_labels = labels[valid_indices]
        filtered_targets.append({'boxes': filtered_boxes, 'labels': filtered_labels})
    return filtered_targets

# Function to evaluate the model
def evaluate_model(model, data_loader):
    model.eval()
    iou_list, loss_list = [], []
    correct_predictions, total_predictions = 0, 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = [{ 'boxes': torch.tensor([obj['bbox'] for obj in t], dtype=torch.float32).to(device),
                         'labels': torch.tensor([obj['category_id'] for obj in t], dtype=torch.int64).to(device)}
                       for t in targets]
            for target in targets:
                boxes = target['boxes']
                boxes[:, 2] += boxes[:, 0]
                boxes[:, 3] += boxes[:, 1]
                target['boxes'] = boxes

            targets = filter_and_correct_boxes(targets)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            for i, output in enumerate(outputs):
                pred_boxes = output['boxes']
                true_boxes = targets[i]['boxes']
                if pred_boxes.size(0) == 0 or true_boxes.size(0) == 0:
                    continue
                iou = box_iou(pred_boxes, true_boxes)
                iou_list.append(iou.mean().item())

                pred_labels = output['labels']
                true_labels = targets[i]['labels']

                # Ensure both tensors are the same size for comparison
                min_size = min(len(pred_labels), len(true_labels))
                correct_predictions += (pred_labels[:min_size] == true_labels[:min_size]).sum().item()
                total_predictions += min_size

    mean_iou = sum(iou_list) / len(iou_list) if iou_list else 0
    accuracy = correct_predictions / total_predictions if total_predictions else 0
    print(f"Mean IoU: {mean_iou:.4f}, Accuracy: {accuracy:.4f}")
    return mean_iou, accuracy

# Load the test dataset for evaluation
test_dataset = CocoDetection(root='/content/swag', annFile='/content/swag/train.json', transform=transform)
test_data_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn)

# Training loop
num_epochs = 10
best_mean_iou = 0

for epoch in range(num_epochs):
    model.train()
    for i, (images, targets) in enumerate(data_loader):
        images = [image.to(device) for image in images]
        targets = [{ 'boxes': torch.tensor([obj['bbox'] for obj in t], dtype=torch.float32).to(device),
                     'labels': torch.tensor([obj['category_id'] for obj in t], dtype=torch.int64).to(device)}
                   for t in targets]
        for target in targets:
            boxes = target['boxes']
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]
            target['boxes'] = boxes

        targets = filter_and_correct_boxes(targets)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(data_loader)}], Loss: {losses.item():.4f}")

    lr_scheduler.step()

    mean_iou, accuracy = evaluate_model(model, test_data_loader)
    if mean_iou > best_mean_iou:
        best_mean_iou = mean_iou
        torch.save(model.state_dict(), 'best_model.pth')
        print("Best model saved")

print("Training completed.")
