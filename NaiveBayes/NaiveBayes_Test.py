# NaiveBayes_Test.py
import os
import joblib
import re
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import seaborn as sns  # for better confusion matrix visualization

# ===============================
# CONFIGURATION
# ===============================
data_dir = os.path.join("..", "Audio_Deepfake_Spectogram")
checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

batch_size = 16
num_classes = 2

# ===============================
# DATA TRANSFORMS
# ===============================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# ===============================
# LOAD TEST DATA
# ===============================
def load_dataset(split):
    dataset = datasets.ImageFolder(os.path.join(data_dir, split), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    X, y = [], []
    for inputs, labels in tqdm(loader, desc=f"Loading {split}"):
        inputs = inputs.view(inputs.size(0), -1).numpy()
        X.append(inputs)
        y.append(labels.numpy())
    X = np.concatenate(X)
    y = np.concatenate(y)
    return X, y, dataset.classes

print("📂 Loading test dataset...")
X_test, y_test, class_names = load_dataset("test")
print(f"✅ Test dataset loaded: {X_test.shape}, Classes: {class_names}")

# ===============================
# FIND LATEST CHECKPOINT
# ===============================
def get_latest_checkpoint():
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".joblib")]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
    return os.path.join(checkpoint_dir, checkpoints[-1])

# ===============================
# LOAD MODEL
# ===============================
latest_checkpoint = get_latest_checkpoint()
if latest_checkpoint is None:
    raise FileNotFoundError("❌ No checkpoint found. Please train the model first.")
print(f"📦 Loading model from: {latest_checkpoint}")
model = joblib.load(latest_checkpoint)

# ===============================
# TEST MODEL
# ===============================
print("🧠 Testing model...")
y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"✅ Test Accuracy: {test_acc:.4f}")

# ===============================
# CONFUSION MATRIX & REPORT
# ===============================
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=class_names)

print("\n📊 Confusion Matrix:")
print(cm)
print("\n🧾 Classification Report:")
print(report)

# ===============================
# SAVE RESULTS
# ===============================
results_dir = os.getcwd()
cm_img_path = os.path.join(results_dir, "confusion_matrix.png")
report_path = os.path.join(results_dir, "classification_report.txt")

# ---- Save Confusion Matrix as Image ----
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix - Naive Bayes")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(cm_img_path)
plt.close()
print(f"💾 Confusion matrix image saved to: {cm_img_path}")

# ---- Save Classification Report as Text ----
with open(report_path, "w") as f:
    f.write("Classification Report - Naive Bayes\n")
    f.write("===================================\n\n")
    f.write(report)

print(f"💾 Classification report saved to: {report_path}")
print("🎯 Testing complete.")
