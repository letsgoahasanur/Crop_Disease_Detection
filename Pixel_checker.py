import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt


normalization = transforms.Compose([
transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                     [0.229, 0.224, 0.225])

])

imgpath=datasets.ImageFolder(root="C:/Users/RAFI/Desktop/Books/Summer 25/Cse 445/Project/PlantVillage",transform=normalization)
dataloader = DataLoader(imgpath, batch_size=32, shuffle=True)









