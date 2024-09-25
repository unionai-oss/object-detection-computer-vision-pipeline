"""
Object Detection Workflow with Flyte and PyTorch using the Faster R-CNN model

note: This Flyte workflow can be broken out into modular tasks for better organization and reusability!
"""
# %%
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torchvision
from flytekit import task, workflow, ImageSpec, Resources, current_context, Deck, Secret
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.models.detection.faster_rcnn import \
    FasterRCNN_ResNet50_FPN_Weights
from torchvision.ops import box_iou
from torchvision.transforms import transforms as T
from flytekit.types.directory import FlyteDirectory
from flytekit.types.file import FlyteFile
import base64
from textwrap import dedent
from pathlib import Path

from datasets import load_dataset
import os
import requests
from dotenv import load_dotenv

load_dotenv()

image = ImageSpec(
    packages=[
        "union==0.1.64",
        "torch",
        "torchvision",
        "matplotlib",
        "pycocotools",
        "datasets",
        "huggingface_hub",
        "python-dotenv",
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
    "num_epochs": 1,
}


# %% ------------------------------
# Data loading helper functions
# --------------------------------

# Convert images to base64 and embed in HTML
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def collate_fn(batch):
    return tuple(zip(*batch))

def dataset_dataloader(
                    root: str,
                    annFile: str,
                    batch_size=2,
                    shuffle=True,
                    num_workers=0,
                ) -> DataLoader:
    # Define the transformations for the images
    transform = T.Compose([T.ToTensor()])

    annFile_path = os.path.join(str(root), annFile)
    print(f"Annotation file path: {annFile_path}")  # Debugging the annotation file path


    # Load the dataset
    dataset = CocoDetection(root=root, annFile=annFile_path, transform=transform)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return data_loader


# %% ------------------------------
# Download dataset - task
# --------------------------------
@task(container_image=image,
      enable_deck=True,
      cache=True,
      cache_version="1.2",
      requests=Resources(cpu="2", mem="2Gi")) 

def download_hf_dataset(repo_id: str = 'sagecodes/union_swag_coco',
                        local_dir: str = "dataset",
                        sub_folder: str = "swag") -> FlyteDirectory:
    
    from huggingface_hub import snapshot_download

    if local_dir:
        dataset_dir = os.path.join(local_dir)
        os.makedirs(dataset_dir, exist_ok=True)

    # Download the dataset repository
    repo_path = snapshot_download(repo_id=repo_id, 
                                  repo_type="dataset",
                                  local_dir=local_dir)
    if sub_folder:
        repo_path = os.path.join(repo_path, sub_folder)
        # use sub_folder to return a specific folder from the dataset

    print(f"Dataset downloaded to {repo_path}")

    print(f"Files in dataset directory: {os.listdir(repo_path)}")

    return FlyteDirectory(repo_path)

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

# %% ------------------------------
# donwload model - task
# --------------------------------
@task(container_image=image,
    cache=True,
    cache_version="1.1",
    requests=Resources(cpu="2", mem="2Gi"))
def download_model() -> torch.nn.Module:
    # Load a pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1, weights_only=True
    )

    return model

# %% ------------------------------
# train model - task
# --------------------------------
@task(container_image=image,
    requests=Resources(cpu="2", mem="8Gi", gpu="1"))
def train_model(model: torch.nn.Module, dataset_dir: FlyteDirectory, num_epochs :int=3) -> torch.nn.Module:

    # TODO: make from dict
    num_classes = 3  # number of classes + background (TODO: add one for the background class automatically)
    num_epochs = num_epochs
    best_mean_iou = 0
    model_dir = "models"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_dir.download()

    os.makedirs(model_dir, exist_ok=True)

    local_dataset_dir = dataset_dir.path  # Use the local path for FlyteDirectory

    data_loader = dataset_dataloader(root=local_dataset_dir, annFile="train.json")
    test_data_loader = dataset_dataloader(root=local_dataset_dir, annFile="train.json")

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
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pth"))
            print("Best model saved")

    print("Training completed.")
    return model


# %% ------------------------------
# evaluate model - task
# ---------------------------------
@task(container_image=image,
      enable_deck=True,
      requests=Resources(cpu="2", mem="8Gi", gpu="1"))
def evaluate_model(model: torch.nn.Module, dataset_dir: FlyteDirectory) -> str:

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset_dir.download()
    local_dataset_dir = dataset_dir.path
    data_loader = dataset_dataloader(root=local_dataset_dir, 
                                     annFile="train.json", shuffle=False)

    model.to(device)
    model.eval()

    num_images = 9  # Number of images to display in the grid
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))  # Create a 3x3 grid
    axes = axes.flatten()  # Flatten the axes array for easier iteration

    iou_list, accuracy_list = [], []
    report = []  # To store the IoU and accuracy report for each image
    global_image_index = 0  # Global image counter across batches
    images_plotted = 0  # Counter for images plotted in the grid

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
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
                boxes[:, 2] += boxes[:, 0]  # Convert width to x_max
                boxes[:, 3] += boxes[:, 1]  # Convert height to y_max
                target["boxes"] = boxes

            outputs = model(images)

            for i, output in enumerate(outputs):
                pred_boxes = output["boxes"]
                true_boxes = targets[i]["boxes"]

                # Get the global image index
                image_index = global_image_index + i

                if pred_boxes.size(0) == 0 or true_boxes.size(0) == 0:
                    report.append(f"Image {image_index}: No valid predictions or ground truths")
                    continue

                # Calculate IoU
                iou = box_iou(pred_boxes, true_boxes)
                mean_iou = iou.mean().item()
                iou_list.append(mean_iou)

                # Calculate accuracy
                pred_labels = output["labels"]
                true_labels = targets[i]["labels"]
                min_size = min(len(pred_labels), len(true_labels))
                correct_predictions = (pred_labels[:min_size] == true_labels[:min_size]).sum().item()
                accuracy = correct_predictions / min_size if min_size else 0
                accuracy_list.append(accuracy)

                # Append report for this image
                report.append(f"Image {image_index}: IoU = {mean_iou:.4f}, Accuracy = {accuracy:.4f}")

                # Plot images (limit to num_images)
                if images_plotted < num_images:
                    img = images[i].cpu().permute(1, 2, 0)  # Convert image to HWC format for plotting
                    ax = axes[images_plotted]  # Access the correct subplot

                    ax.imshow(img)
                    for j in range(len(output['boxes'])):
                        bbox = output['boxes'][j].cpu().numpy()
                        score = output['scores'][j].cpu().item()
                        label = output['labels'][j].cpu().item()

                        if score > 0.6:  # Only display predictions with confidence score above 0.6
                            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                                     linewidth=2, edgecolor='r', facecolor='none')
                            ax.add_patch(rect)
                            ax.text(bbox[0], bbox[1], f"{label}: {score:.2f}", color="white", fontsize=8,
                                    bbox=dict(facecolor="red", alpha=0.5))
                    ax.axis('off')  # Hide axes
                    images_plotted += 1

            # Update global image index after processing the batch
            global_image_index += len(images)

            if images_plotted >= num_images:  # Break once we've plotted 9 images
                break

    # Compute overall metrics
    overall_iou = sum(iou_list) / len(iou_list) if iou_list else 0
    overall_accuracy = sum(accuracy_list) / len(accuracy_list) if accuracy_list else 0



    # Save the image grid
    pred_boxes_imgs = "prediction_grid.png"
    plt.tight_layout()
    plt.savefig(pred_boxes_imgs)
    plt.close()

    train_image_base64 = image_to_base64(pred_boxes_imgs)

    # Prepare the report as text
    report_text = "\n".join(report)
    overall_report = dedent(f"""
    Overall Metrics:
    ----------------
    Mean IoU: {overall_iou:.4f}
    Mean Accuracy: {overall_accuracy:.4f}

    Per-Image Metrics:
    ------------------
    {report_text}
    """)

    # Display the report in FlyteDeck
    ctx = current_context()
    deck = Deck("Evaluation Results")
    html_report = dedent(f"""
    <div style="font-family: Arial, sans-serif; line-height: 1.6;">
       <h2 style="color: #2C3E50;">Predicted Bounding Boxes</h2>
        <img src="data:image/png;base64,{train_image_base64}" width="600">
    </div>               
    <div style="font-family: Arial, sans-serif; line-height: 1.6;">
        <h2 style="color: #2C3E50;">Evaluation Report</h2>
        <pre>{overall_report}</pre>
    </div>

    """)

    # Append HTML content to the deck
    deck.append(html_report)
    ctx.decks.insert(0, deck)

    return overall_report

# %% ------------------------------
# upload model to hub - task
# --------------------------------
@task(
    container_image=image,
    requests=Resources(cpu="2", mem="2Gi"),
    secret_requests=[Secret(group=None, key="hf_token")],
)
def upload_model_to_hub(model: torch.nn.Module, repo_name: str = "sagecodes/cv-object") -> str:
    from huggingface_hub import HfApi
    # Get the Flyte context and define the model path
    ctx = current_context()
    model_path = "best_model.pth"  # Save the model locally as "best_model.pth"

    # Save the model's state dictionary
    torch.save(model.state_dict(), model_path)

    # Set Hugging Face token from local environment or Flyte secrets
    hf_token = os.getenv("HF_TOKEN")
    if hf_token is None:
        # If HF_TOKEN is not found, attempt to get it from the Flyte secrets
        hf_token = ctx.secrets.get(key="hf_token")
        print("Using Hugging Face token from Flyte secrets.")
    else:
        print("Using Hugging Face token from environment variable.")

    # Create a new repository (if it doesn't exist) on Hugging Face Hub
    api = HfApi()
    api.create_repo(repo_name, token=hf_token, exist_ok=True)

    # Upload the model to the Hugging Face repository
    api.upload_file(
        path_or_fileobj=model_path,      # Path to the local file
        path_in_repo="pytorch_model.bin", # Destination path in the repo
        repo_id=repo_name,
        commit_message="Upload Faster R-CNN model",
        token=hf_token
    )

    return f"Model uploaded to Hugging Face Hub: https://huggingface.co/{repo_name}"


# %% ------------------------------
# Object Detection Workflow
# --------------------------------
@workflow
def object_detection_workflow() -> torch.nn.Module:
    dataset_dir = download_hf_dataset(repo_id="sagecodes/union_flyte_swag_object_detection")
    model = download_model()
    trained_model = train_model(model=model, dataset_dir=dataset_dir, num_epochs=1)
    evaluate_model(model=trained_model, dataset_dir=dataset_dir)
    upload_model_to_hub(model=trained_model, repo_name="sagecodes/cv-object")
    return model

# union run --remote workflow2.py object_detection_workflow
# union run workflow2.py object_detection_workflow