# This function loads and preprocesses the plant leaf images for training and testing

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transforms for data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((256, 256)),   # just in case, but your images are 256 already
    transforms.ToTensor(),            # convert PIL image to tensor
    transforms.Normalize([0.485, 0.456, 0.406],   # normalize with ImageNet means/std
                         [0.229, 0.224, 0.225])
])

# Load dataset (replace 'path_to_dataset' with your actual path)
dataset = datasets.ImageFolder(root='C:/Users/RAFI/Desktop/Books/Summer 25/Cse 445/Project/PlantVillage', transform=transform)

# Split dataset into train and test (80% train, 20% test)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Data loaders for batching
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


print(f"Train samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Number of classes: {len(dataset.classes)}")
print(f"Classes: {dataset.classes}")
