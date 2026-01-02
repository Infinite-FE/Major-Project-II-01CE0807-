import os
import joblib
import re
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils import shuffle
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

data_dir = os.path.join("..", "Audio_Deepfake_Spectogram")
checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

batch_size = 16
num_epochs = 10
num_classes = 2

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def load_dataset(split):
    dataset = datasets.ImageFolder(os.path.join(data_dir, split), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    X, y = [], []
    for inputs, labels in tqdm(loader, desc=f"Loading {split}"):
        inputs = inputs.view(inputs.size(0), -1).numpy()
        X.append(inputs)
        y.append(labels.numpy())
    X = np.concatenate(X)
    y = np.concatenate(y)
    return X, y

print("📂 Loading datasets...")
X_train, y_train = load_dataset("train")
X_val, y_val = load_dataset("val")
print(f"✅ Train: {X_train.shape}, Val: {X_val.shape}")

def get_latest_checkpoint():
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".joblib")]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(re.findall(r'\d+', x)[-1]))
    return os.path.join(checkpoint_dir, checkpoints[-1])

def train_random_forest(num_epochs=10):
    latest_checkpoint = get_latest_checkpoint()
    start_epoch = 0

    if latest_checkpoint:
        print(f"🔁 Found checkpoint: {latest_checkpoint}")
        model = joblib.load(latest_checkpoint)
        start_epoch = int(re.findall(r'\d+', latest_checkpoint)[-1])
        print(f"✅ Resuming from epoch {start_epoch + 1}")
    else:
        print("🚀 Starting fresh training...")
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

    for epoch in range(start_epoch, num_epochs):
        try:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 30)

            X_train_, y_train_ = shuffle(X_train, y_train)
            model.fit(X_train_, y_train_)

            train_preds = model.predict(X_train)
            val_preds = model.predict(X_val)
            train_probs = model.predict_proba(X_train)
            val_probs = model.predict_proba(X_val)

            train_acc = accuracy_score(y_train, train_preds)
            val_acc = accuracy_score(y_val, val_preds)
            train_loss = log_loss(y_train, train_probs, labels=[0, 1])
            val_loss = log_loss(y_val, val_probs, labels=[0, 1])

            print(f"train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
            print(f"val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            checkpoint_path = os.path.join(checkpoint_dir, f"randomforest_epoch_{epoch + 1}.joblib")
            joblib.dump(model, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")

        except KeyboardInterrupt:
            print("\n⏸️ Training paused by user (Ctrl+C). Saving progress...")
            checkpoint_path = os.path.join(checkpoint_dir, f"randomforest_paused_epoch_{epoch + 1}.joblib")
            joblib.dump(model, checkpoint_path)
            print(f"💾 Paused checkpoint saved: {checkpoint_path}")
            break

    print("🎯 Training complete.")

if __name__ == "__main__":
    train_random_forest(num_epochs=num_epochs)
