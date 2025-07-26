import torch
import torchvision
import matplotlib.pyplot as plt

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
print("Torchvision transform loaded!")

plt.plot([1, 8, 3], [1, 5, 9])
plt.title("Matplotlib Test Plot")
plt.show()
