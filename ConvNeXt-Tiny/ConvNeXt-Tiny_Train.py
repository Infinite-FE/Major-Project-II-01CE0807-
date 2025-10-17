# ConvNeXt-Tiny_Train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import multiprocessing
import re

# ===============================
# CONFIGURATION
# ===============================
data_dir = os.path.join("..", "Audio_Deepfake_Spectogram")

num_classes = 2  # Real / Fake
batch_size = 16
num_epochs = 10
learning_rate = 1e-4
num_workers = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Checkpoints folder inside ConvNeXt-Tiny
checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# ===============================
# DATA TRANSFORMS
# ===============================
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# ===============================
# MODEL SETUP (ConvNeXt-Tiny)
# ===============================
def create_model(num_classes):
    model = models.convnext_tiny(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False  # Freeze all pretrained layers

    # Replace the classifier
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
# TRAINING FUNCTION (with resume)
# ===============================
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=10):
    start_epoch = 0

    # Resume from checkpoint if available
    latest_checkpoint = get_latest_checkpoint()
    if latest_checkpoint:
        print(f"🔁 Found checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Resuming training from epoch {start_epoch + 1}")
    else:
        print("🚀 Starting fresh training...")

    model = model.to(device)

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            loop = tqdm(dataloaders[phase], desc=f"{phase.upper()}", leave=False)
            for inputs, labels in loop:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"convnext-tiny_epoch_{epoch + 1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")

    print("🎯 Training complete.")

# ===============================
# MAIN ENTRY
# ===============================
if __name__ == "__main__":
    multiprocessing.freeze_support()

    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Load datasets
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val', 'test']
    }

    dataloaders = {
        x: DataLoader(image_datasets[x],
                      batch_size=batch_size,
                      shuffle=True if x == 'train' else False,
                      num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())
        for x in ['train', 'val', 'test']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes
    print(f"Classes found: {class_names}")
    print(f"Dataset sizes: {dataset_sizes}")

    # Build model, criterion, optimizer
    model = create_model(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # Train / Resume
    train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=num_epochs)