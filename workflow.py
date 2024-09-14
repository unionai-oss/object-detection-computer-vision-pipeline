# %%
import torch
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.transforms import transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import box_iou

# Check the available device
# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")

# %%

# Define the custom collate function
def collate_fn(batch):
    return tuple(zip(*batch))

# Define the transformations for the images
transform = T.Compose([T.ToTensor()])

# Load the dataset
dataset = CocoDetection(root='data/', annFile='data/train.json', transform=transform)

# Use the custom collate function
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)

# %%
# Function to display an image with its bounding boxes
def show_image_with_boxes(image, annotations):
    fig, ax = plt.subplots(1)
    ax.imshow(image.permute(1, 2, 0))  # Convert image tensor to (H, W, C) format for visualization

    for annotation in annotations:
        bbox = annotation['bbox']
        # Create a rectangle patch
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

# Visualize a batch of images
for images, targets in data_loader:
    for i, image in enumerate(images):
        show_image_with_boxes(image, targets[i])
    break  # Display only the first batch

# %% Download model weights
# Load a pre-trained faster RCNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)



# %% TRAIN MODEL WITH CUSTOM DATASET #########
# Create model for classes
print(model.roi_heads.box_predictor)

# Replace the classifier with a new one (number of classes + 1 for the background)
num_classes = 2  # Example: 1 class (goose) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

print(model.roi_heads.box_predictor)
# %% Training the model
# Move the entire model to the correct device
model.to(device)

# Define optimizer and learning rate
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# number of epochs for training loops
num_epochs = 2


# Function to filter out and correct invalid boxes
# This is an optoinal step if your dataset may contain invalid annotations
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

# %% training loops
for epoch in range(num_epochs):
    model.train()
    i = 0
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{ 'boxes': torch.tensor([obj['bbox'] for obj in t], dtype=torch.float32).to(device),
                     'labels': torch.tensor([obj['category_id'] for obj in t], dtype=torch.int64).to(device)}
                   for t in targets]

        for target in targets:
            boxes = target['boxes']
            # Convert [x, y, width, height] to [x_min, y_min, x_max, y_max]
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            target['boxes'] = boxes

        targets = filter_and_correct_boxes(targets)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # Move targets to device

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(data_loader)}], Loss: {losses.item():.4f}")
        i += 1

    lr_scheduler.step()
     
#%% Save the model
# Save the model's state dictionary
model_save_path = 'models/geese-object-detection-model.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# %% LOAD MODEL #########
# Load the model's state dictionary
model_load_path = 'models/geese-object-detection-model.pth'
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)  # Initialize the model
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)  # Adjust the classifier

model.load_state_dict(torch.load(model_load_path, map_location=device))  # Load the state dictionary
model.to(device)  # Move the model to the appropriate device
print(f"Model loaded from {model_load_path}")



# %% Load Test Data
# Define the transformations for the test images
test_transform = T.Compose([T.ToTensor()])

# Load the test dataset
test_dataset = CocoDetection(root='data/', annFile='data/test.json', transform=test_transform)

# Use the custom collate function for the test dataset
test_data_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)

# Function to display an image with its bounding boxes and labels
def show_image_with_predictions(image, predictions):
    fig, ax = plt.subplots(1)
    ax.imshow(image.permute(1, 2, 0))  # Convert image tensor to (H, W, C) format for visualization

    for prediction in predictions:
        bbox = prediction['bbox']
        score = prediction['score']
        label = prediction['label']
        # Create a rectangle patch
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(bbox[0], bbox[1], f"{label}: {score:.2f}", color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

    plt.show()

# Visualize predictions on the test dataset
model.eval()
with torch.no_grad():
    for images, targets in test_data_loader:
        images = list(image.to(device) for image in images)

        outputs = model(images)

        for i, output in enumerate(outputs):
            image = images[i].cpu()
            predictions = []
            for j in range(len(output['boxes'])):
                bbox = output['boxes'][j].cpu().numpy()
                score = output['scores'][j].cpu().item()
                label = output['labels'][j].cpu().item()
                if score > 0.5:  # Only display predictions with a confidence score above 0.5
                    predictions.append({'bbox': bbox, 'score': score, 'label': label})
            # show_image_with_predictions(image, predictions)
            show_image_with_boxes(image, predictions)
            



# %% EVALUATE MODEL #########
test_dataset = CocoDetection(root='data/', annFile='data/test.json', transform=test_transform)
# Evaluation
model.eval()
with torch.no_grad():
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{ 'boxes': torch.tensor([obj['bbox'] for obj in t], dtype=torch.float32).to(device),
                     'labels': torch.tensor([obj['category_id'] for obj in t], dtype=torch.int64).to(device)}
                   for t in targets]

        for target in targets:
            boxes = target['boxes']
            # Convert [x, y, width, height] to [x_min, y_min, x_max, y_max]
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
            target['boxes'] = boxes

        targets = filter_and_correct_boxes(targets)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # Move targets to device

        outputs = model(images)

        for i, output in enumerate(outputs):
            pred_boxes = output['boxes']
            true_boxes = targets[i]['boxes']
            iou = box_iou(pred_boxes, true_boxes)
            print(f"IOU: {iou.mean().item():.4f}")





# %%
