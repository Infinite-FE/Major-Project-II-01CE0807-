import os
import joblib
import re
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils import shuffle
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

# ===============================
# CONFIGURATION
# ===============================
data_dir = os.path.join("..", "Audio_Deepfake_Spectogram")
checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

batch_size = 16
num_epochs = 10
num_classes = 2

# ===============================
# DATA TRANSFORMS
# ===============================
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# ===============================
# LOAD DATA
# ===============================
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

# ===============================
# DIMENSIONALITY REDUCTION (⚡ speed-up)
# ===============================
print("⚙️ Applying PCA for dimensionality reduction...")
pca = PCA(n_components=100)  # reduce features to 100 dims
X_train = pca.fit_transform(X_train)
X_val = pca.transform(X_val)
print(f"✅ Reduced to shape: {X_train.shape}")

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
# TRAINING FUNCTION
# ===============================
def train_knn(num_epochs=10):
    latest_checkpoint = get_latest_checkpoint()
    start_epoch = 0

    if latest_checkpoint:
        print(f"🔁 Found checkpoint: {latest_checkpoint}")
        model = joblib.load(latest_checkpoint)
        start_epoch = int(re.findall(r'\d+', latest_checkpoint)[-1])
        print(f"✅ Resuming from epoch {start_epoch + 1}")
    else:
        print("🚀 Starting fresh training...")
        model = KNeighborsClassifier(n_neighbors=3, n_jobs=-1, weights='distance')

    for epoch in range(start_epoch, num_epochs):
        try:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 30)

            # train on random subset (⚡ optional speed-up)
            sample_size = min(4000, len(X_train))  # train on max 4000 samples
            idx = np.random.choice(len(X_train), sample_size, replace=False)
            X_train_, y_train_ = shuffle(X_train[idx], y_train[idx])

            model.fit(X_train_, y_train_)

            train_preds = model.predict(X_train_)
            val_preds = model.predict(X_val)
            train_probs = model.predict_proba(X_train_)
            val_probs = model.predict_proba(X_val)

            train_acc = accuracy_score(y_train_, train_preds)
            val_acc = accuracy_score(y_val, val_preds)
            train_loss = log_loss(y_train_, train_probs, labels=[0, 1])
            val_loss = log_loss(y_val, val_probs, labels=[0, 1])

            print(f"train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
            print(f"val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

            checkpoint_path = os.path.join(checkpoint_dir, f"knn_epoch_{epoch + 1}.joblib")
            joblib.dump(model, checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")

        except KeyboardInterrupt:
            print("\n⏸️ Training paused by user (Ctrl+C). Saving progress...")
            checkpoint_path = os.path.join(checkpoint_dir, f"knn_paused_epoch_{epoch + 1}.joblib")
            joblib.dump(model, checkpoint_path)
            print(f"💾 Paused checkpoint saved: {checkpoint_path}")
            break

    print("🎯 Training complete.")

if __name__ == "__main__":
    train_knn(num_epochs=num_epochs)