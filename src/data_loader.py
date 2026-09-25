import os
import kagglehub
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

def get_data_dir(local_data_dir="data"):
    if os.path.exists(os.path.join(local_data_dir, "train")):
        return local_data_dir
    elif os.path.exists(os.path.join(local_data_dir, "chest_xray", "train")):
        return os.path.join(local_data_dir, "chest_xray")
    
    print("Local data not found. Downloading via kagglehub...")
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    
    if os.path.exists(os.path.join(path, "chest_xray", "train")):
        return os.path.join(path, "chest_xray")
    return path

def load_data(batch_size=32, target_size=(150, 150)):
    data_dir = get_data_dir()
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    print(f"Loading data from: {data_dir}")

    train_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    # Class balancing
    class_counts = [0, 0]
    for _, target in train_dataset.samples:
        class_counts[target] += 1
    
    total_samples = sum(class_counts)
    class_weights = [total_samples / (2 * c) for c in class_counts]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, class_weights_tensor, train_dataset.classes
