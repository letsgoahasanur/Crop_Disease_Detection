from torchvision import datasets, transforms
from torch.utils.data import random_split

# Image transformation (must match training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Your dataset path
dataset_path = r'C:\Users\RAFI\Desktop\Books\Summer 25\Cse 445\Project\PlantVillage'

# Load the full dataset
full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# Split into train and test
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# View a sample path from training
train_index = train_dataset.indices[0]
print("Sample Train Image Path:", full_dataset.samples[train_index][0])

# View a sample path from testing
test_index = test_dataset.indices[0]
print("Sample Test Image Path:", full_dataset.samples[test_index][0])
