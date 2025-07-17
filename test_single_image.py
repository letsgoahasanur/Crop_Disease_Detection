import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# 1. Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Define SAME model as training
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# 3. Load model + weights
model = SimpleCNN(num_classes=15).to(device)
model.load_state_dict(torch.load(r'C:\Users\RAFI\PycharmProjects\Crop_detection\model.pth', map_location=device))
model.eval()

# 4. Image transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # match training
])

# 5. Load and preprocess the image
img_path = r"C:\Users\RAFI\Desktop\Books\Summer 25\Cse 445\Project\PlantVillage\Tomato__Tomato_YellowLeaf__Curl_Virus\64009556-6711-4fcd-9235-17d74152dc0f___YLCV_GCREC 2659.JPG"
image = Image.open(img_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0).to(device)

# 6. Predict
with torch.no_grad():
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)

# 7. Class names
class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
               'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
               'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
               'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
               'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

# 8. Output
print(f"✅ Predicted Class: {class_names[predicted.item()]}")
