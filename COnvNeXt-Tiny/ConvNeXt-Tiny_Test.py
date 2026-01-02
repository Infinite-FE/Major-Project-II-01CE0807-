# ConvNeXt-Tiny_Test.py
import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import re

# ===============================
# CONFIGURATION
# ===============================
data_dir = os.path.join("..", "Audio_Deepfake_Spectogram")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
num_workers = 0
num_classes = 2  # Real / Fake

# Checkpoints folder inside ConvNeXt-Tiny
checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")

# ===============================
# DATA TRANSFORMS
# ===============================
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===============================
# LOAD TEST DATA
# ===============================
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=torch.cuda.is_available())
class_names = test_dataset.classes
print(f"Classes found: {class_names}")

# ===============================
# MODEL SETUP (ConvNeXt-Tiny)
# ===============================
def create_model(num_classes):
    model = models.convnext_tiny(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False  # Freeze pretrained layers
    num_ftrs = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_ftrs, num_classes)
    return model

# ===============================
# FIND LATEST CHECKPOINT
# ===============================
def get_latest_checkpoint():
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))  # sort by epoch number
    return os.path.join(checkpoint_dir, checkpoints[-1])

# ===============================
# TEST FUNCTION
# ===============================
def test_model(model, dataloader, class_names):
    model.eval()
    model = model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Confusion Matrix and Classification Report
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)

    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", cm)

    # ===============================
    # SAVE CLASSIFICATION REPORT
    # ===============================
    report_path = os.path.join(os.getcwd(), "convnext_tiny_classification_report.txt")
    with open(report_path, "w") as f:
        f.write("=== ConvNeXt-Tiny Classification Report ===\n\n")
        f.write(report)
    print(f"📄 Classification report saved at: {report_path}")

    # ===============================
    # SAVE CONFUSION MATRIX AS IMAGE
    # ===============================
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix - ConvNeXt-Tiny")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    cm_image_path = os.path.join(os.getcwd(), "convnext_tiny_confusion_matrix.png")
    plt.savefig(cm_image_path)
    plt.close()
    print(f"🖼️ Confusion matrix image saved at: {cm_image_path}")

# ===============================
# MAIN ENTRY
# ===============================
if __name__ == "__main__":
    # Build model
    model = create_model(num_classes)

    # Load latest checkpoint
    latest_checkpoint = get_latest_checkpoint()
    if latest_checkpoint:
        print(f"🔁 Loading checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Model weights loaded successfully.")
    else:
        print("⚠️ No checkpoint found! Please train the model first.")
        exit()

    # Run testing
    print("\n🚀 Starting testing on test dataset...")
    test_model(model, test_loader, class_names)
    print("\n✅ Testing complete.")