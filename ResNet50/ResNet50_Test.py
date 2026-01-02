import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

# ===============================
# CONFIGURATION
# ===============================
data_dir = os.path.join("..", "Audio_Deepfake_Spectogram")
checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 16
num_workers = 0
num_classes = 2  # Real / Fake

# ===============================
# DATA TRANSFORMS
# ===============================
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===============================
# LOAD TEST DATASET
# ===============================
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=torch.cuda.is_available())
class_names = test_dataset.classes
print(f"Classes found: {class_names}")

# ===============================
# MODEL SETUP
# ===============================
def create_model(num_classes):
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# ===============================
# FIND LATEST CHECKPOINT
# ===============================
def get_latest_checkpoint():
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
    return os.path.join(checkpoint_dir, checkpoints[-1])

# ===============================
# TEST FUNCTION
# ===============================
def test_model(model, dataloader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return np.array(y_true), np.array(y_pred)

# ===============================
# MAIN EXECUTION
# ===============================
if __name__ == "__main__":
    latest_checkpoint = get_latest_checkpoint()
    if latest_checkpoint is None:
        print("❌ No checkpoint found in 'checkpoints' folder.")
        exit()

    print(f"🔁 Loading checkpoint: {latest_checkpoint}")
    model = create_model(num_classes)
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print("🚀 Running inference on test set...")
    y_true, y_pred = test_model(model, test_loader)

    # ===============================
    # CONFUSION MATRIX & REPORT
    # ===============================
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\n📊 Classification Report:\n", report)
    print("✅ Confusion Matrix:\n", cm)

    # --- Plot confusion matrix ---
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('ResNet50 Confusion Matrix')

    # Save the image where this script is located
    save_path = os.path.join(os.getcwd(), "ResNet50_ConfusionMatrix.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"🖼️ Confusion matrix saved at: {save_path}")

    # Also save classification report as text
    report_path = os.path.join(os.getcwd(), "ResNet50_ClassificationReport.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"📝 Classification report saved at: {report_path}")