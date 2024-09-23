"""
Object Detection Workflow with Flyte and PyTorch using the Faster R-CNN model

"""
# %%
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torchvision
from flytekit import task, workflow, ImageSpec, Resources
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.models.detection.faster_rcnn import \
    FasterRCNN_ResNet50_FPN_Weights
from torchvision.ops import box_iou
from torchvision.transforms import transforms as T
from flytekit.types.directory import FlyteDirectory


from datasets import load_dataset
import os
import requests


image = ImageSpec(
    packages=[
        "union==0.1.64",
        "torch",
        "torchvision",
        "matplotlib",
        "pycocotools",
        "datasets",
    ],
)

# Check and set the available device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

hyperparams = {
    "batch_size": 2,
    "num_workers": 4,
    "lr": 0.005,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "step_size": 3,
    "gamma": 0.1,
    "num_epochs": 10,
}


# %% ------------------------------
# Load data set (create data loader)
# --------------------------------
# @task
# Define the custom collate function
def collate_fn(batch):
    return tuple(zip(*batch))

def dataset_dataloader(
    root,
    annFile,
    batch_size=2,
    shuffle=True,
    num_workers=0,
) -> DataLoader:
    # Define the transformations for the images
    transform = T.Compose([T.ToTensor()])

    # Load the dataset
    # dataset = CocoDetection(root='data/swag', annFile='data/swag/train.json', transform=transform)
    # data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
    dataset = CocoDetection(root=root, annFile=annFile, transform=transform)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return data_loader


# %% ------------------------------
# Download data set - task
# --------------------------------
@task(container_image=image, 
      requests=Resources(cpu="2", mem="2Gi")) 

def download_hf_dataset(repo_id: str = 'sagecodes/union_swag_coco',
                        local_dir: str = '',
                        sub_folder: str = None) -> FlyteDirectory:
    from huggingface_hub import snapshot_download

    # Download the dataset repository
    repo_path = snapshot_download(repo_id=repo_id, 
                                  repo_type="dataset",
                                  local_dir=local_dir)
    if sub_folder:
        repo_path = os.path.join(repo_path, sub_folder)

    print(f"Dataset downloaded to {repo_path}")

    return FlyteDirectory(repo_path)

# download_hf_dataset(sub_folder="swag")

# %% ------------------------------
# visualize data - task
# --------------------------------
@task(container_image=image,
    requests=Resources(cpu="2", mem="2Gi"))
def visualize_data():
    data_loader = dataset_dataloader("data/swag", "data/swag/train.json")

    # Function to display an image with its bounding boxes
    def show_image_with_boxes(image, annotations):
        fig, ax = plt.subplots(1)
        ax.imshow(image.permute(1, 2, 0))

        for annotation in annotations:
            bbox = annotation["bbox"]
            rect = patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2],
                bbox[3],
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax.add_patch(rect)

        plt.show()

    # Visualize a batch of images
    for images, targets in data_loader:
        for i, image in enumerate(images):
            show_image_with_boxes(image, targets[i])


# visualize_data()


# %% ------------------------------
# donwload model - task
# --------------------------------
@task(container_image=image,
    requests=Resources(cpu="2", mem="2Gi"))
def download_model() -> torch.nn.Module:
    # Load a pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    )
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # model.to(device)
    # print(model)
    return model


# model = download_model()
# print(model)
# print(model.roi_heads.box_predictor.cls_score.in_features)


# %% ------------------------------
# train model - task
# --------------------------------
@task(container_image=image,
    requests=Resources(cpu="2", mem="4Gi", gpu="1"))
def train_model(model: torch.nn.Module, data_loader: DataLoader, hyperparams: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_classes = 3  # number of classes + background (TODO: add one for the background class automatically)
    num_epochs = 1
    best_mean_iou = 0

    # Modify the model to add a new classification head based on the number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
    )

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
            boxes = target["boxes"]
            labels = target["labels"]
            valid_indices = []
            for i, box in enumerate(boxes):
                if box[2] > box[0] and box[3] > box[1]:
                    valid_indices.append(i)
                else:
                    print(f"Invalid box found and removed: {box}")
            filtered_boxes = boxes[valid_indices]
            filtered_labels = labels[valid_indices]
            filtered_targets.append(
                {"boxes": filtered_boxes, "labels": filtered_labels}
            )
        return filtered_targets

    # Function to evaluate the model
    def evaluate_model(model, data_loader):
        model.eval()
        iou_list, loss_list = [], []
        correct_predictions, total_predictions = 0, 0
        with torch.no_grad():
            for images, targets in data_loader:
                images = [image.to(device) for image in images]
                targets = [
                    {
                        "boxes": torch.tensor(
                            [obj["bbox"] for obj in t], dtype=torch.float32
                        ).to(device),
                        "labels": torch.tensor(
                            [obj["category_id"] for obj in t], dtype=torch.int64
                        ).to(device),
                    }
                    for t in targets
                ]
                for target in targets:
                    boxes = target["boxes"]
                    boxes[:, 2] += boxes[:, 0]
                    boxes[:, 3] += boxes[:, 1]
                    target["boxes"] = boxes

                targets = filter_and_correct_boxes(targets)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                outputs = model(images)

                for i, output in enumerate(outputs):
                    pred_boxes = output["boxes"]
                    true_boxes = targets[i]["boxes"]
                    if pred_boxes.size(0) == 0 or true_boxes.size(0) == 0:
                        continue
                    iou = box_iou(pred_boxes, true_boxes)
                    iou_list.append(iou.mean().item())

                    pred_labels = output["labels"]
                    true_labels = targets[i]["labels"]

                    # Ensure both tensors are the same size for comparison
                    min_size = min(len(pred_labels), len(true_labels))
                    correct_predictions += (
                        (pred_labels[:min_size] == true_labels[:min_size]).sum().item()
                    )
                    total_predictions += min_size

        mean_iou = sum(iou_list) / len(iou_list) if iou_list else 0
        accuracy = correct_predictions / total_predictions if total_predictions else 0
        print(f"Mean IoU: {mean_iou:.4f}, Accuracy: {accuracy:.4f}")
        return mean_iou, accuracy

    # Load the test dataset for evaluation

    transform = T.Compose([T.ToTensor()])
    test_dataset = CocoDetection(
        root="data/swag", annFile="data/swag/train.json", transform=transform
    )
    test_data_loader = DataLoader(
        test_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    for epoch in range(num_epochs):
        model.train()
        for i, (images, targets) in enumerate(data_loader):
            images = [image.to(device) for image in images]
            targets = [
                {
                    "boxes": torch.tensor(
                        [obj["bbox"] for obj in t], dtype=torch.float32
                    ).to(device),
                    "labels": torch.tensor(
                        [obj["category_id"] for obj in t], dtype=torch.int64
                    ).to(device),
                }
                for t in targets
            ]
            for target in targets:
                boxes = target["boxes"]
                boxes[:, 2] += boxes[:, 0]
                boxes[:, 3] += boxes[:, 1]
                target["boxes"] = boxes

            targets = filter_and_correct_boxes(targets)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if i % 10 == 0:
                print(
                    f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(data_loader)}], Loss: {losses.item():.4f}"
                )

        lr_scheduler.step()

        mean_iou, accuracy = evaluate_model(model, test_data_loader)
        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved")

    print("Training completed.")


# train_model(
#     download_model(), dataset_dataloader("data/swag", "data/swag/train.json"), hyperparams
# )


# %% ------------------------------
# evaluate model - task
# --------------------------------

@task(container_image=image,
    requests=Resources(cpu="2", mem="2Gi", gpu="1"))
def evaluate_model(model: torch.nn.Module, data_loader: DataLoader):

    num_classes = 3  # number of classes + background (TODO: add one for the background class automatically)
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Load the model's state dictionary
    model_load_path = "best_model.pth"
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=False
    )  # Initialize the model
    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, num_classes
        )
    )  # Adjust the classifier

    model.load_state_dict(
        torch.load(model_load_path, map_location=device)
    )  # Load the state dictionary
    model.to(device)  # Move the model to the appropriate device
    print(f"Model loaded from {model_load_path}")

    # Define the transformations for the test images
    test_transform = T.Compose([T.ToTensor()])

    # Load the test dataset
    test_dataset = CocoDetection(
        root="data/swag/", annFile="data/swag/train.json", transform=test_transform
    )

    # Use the custom collate function for the test dataset
    test_data_loader = DataLoader(
        test_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn
    )

    # Function to display an image with its bounding boxes and labels
    def show_image_with_predictions(image, predictions):
        fig, ax = plt.subplots(1)
        ax.imshow(
            image.permute(1, 2, 0)
        )  # Convert image tensor to (H, W, C) format for visualization

        for prediction in predictions:
            bbox = prediction["bbox"]
            score = prediction["score"]
            label = prediction["label"]
            # Create a rectangle patch
            rect = patches.Rectangle(
                (bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1],
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            # Add the patch to the Axes
            ax.add_patch(rect)
            plt.text(
                bbox[0],
                bbox[1],
                f"{label}: {score:.2f}",
                color="white",
                fontsize=8,
                bbox=dict(facecolor="red", alpha=0.5),
            )

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
                for j in range(len(output["boxes"])):
                    bbox = output["boxes"][j].cpu().numpy()
                    score = output["scores"][j].cpu().item()
                    label = output["labels"][j].cpu().item()
                    if (
                        score > 0.5
                    ):  # Only display predictions with a confidence score above 0.5
                        predictions.append(
                            {"bbox": bbox, "score": score, "label": label}
                        )
                show_image_with_predictions(image, predictions)


# evaluate_model(download_model(), dataset_dataloader("data/swag", "data/swag/train.json"))
# add IoU calculation for each photo and only show a few images in Flytedeck
# %%
@workflow
def object_detection_workflow():
    model = download_model()
    data_loader = dataset_dataloader("data/swag", "data/swag/train.json")
    train_model(model=model, data_loader=data_loader, hyperparams=hyperparams)
    evaluate_model(model=model, data_loader=data_loader)

# union run --remote workflow2.py object_detection_workflow

# union run --remote --copy-all workflow2.py object_detection_workflow
