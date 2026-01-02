# MobileNetV3_Test.py
import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re

# ===============================
# CONFIGURATION
# ===============================
data_dir = os.path.join("..", "Audio_Deepfake_Spectogram")
checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
output_dir = os.path.join(os.getcwd(), "results")
os.makedirs(output_dir, exist_ok=True)

batch_size = 16
num_classes = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
# LOAD TEST DATASET
# ===============================
print("📂 Loading test dataset...")
test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
class_names = test_dataset.classes
print(f"✅ Found {len(test_dataset)} test samples, Classes: {class_names}")

# ===============================
# MODEL SETUP
# ===============================
def create_model(num_classes):
    model = models.mobilenet_v3_large(pretrained=False)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, num_classes)
    return model

# ===============================
# LOAD LATEST CHECKPOINT
# ===============================
def get_latest_checkpoint():
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not checkpoints:
        raise FileNotFoundError("❌ No checkpoint file found in 'checkpoints' directory!")
    checkpoints.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))  # sort by epoch number
    return os.path.join(checkpoint_dir, checkpoints[-1])

latest_checkpoint = get_latest_checkpoint()
print(f"🔍 Loading model checkpoint: {latest_checkpoint}")

# Load model and checkpoint
model = create_model(num_classes)
checkpoint = torch.load(latest_checkpoint, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# ===============================
# EVALUATION
# ===============================
all_preds = []
all_labels = []

print("⚙️ Running evaluation on test data...")
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ===============================
# METRICS & RESULTS
# ===============================
print("📊 Generating confusion matrix and classification report...")

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
acc = np.trace(cm) / np.sum(cm)
print(f"✅ Test Accuracy: {acc:.4f}")

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title(f"Confusion Matrix (Accuracy: {acc:.2%})")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
cm_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()
print(f"🖼️ Confusion matrix saved to: {cm_path}")

# Classification Report
report = classification_report(all_labels, all_preds, target_names=class_names)
report_path = os.path.join(output_dir, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(report)
print(f"📄 Classification report saved to: {report_path}")

# ===============================
# SUMMARY
# ===============================
print("\n🎯 Evaluation complete!")
print(f"Accuracy: {acc:.4f}")
print(f"Results saved in folder: {output_dir}")