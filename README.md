# 🍅 Crop Disease Detection using CNN

This project detects plant leaf diseases using a Convolutional Neural Network (CNN). It is trained on images from the PlantVillage dataset.

## 📂 Features
- Classifies 15 different types of crop diseases and healthy leaves.
- Model trained using PyTorch.
- Single-image prediction with `.pth` model support.

## 🧠 Model Info
- Input image size: 128x128
- Architecture: Simple CNN with Conv2D and Fully Connected layers
- Trained on: Tomato, Potato, and Pepper images

## 🖼️ Sample Classes
- `Tomato__Tomato_YellowLeaf__Curl_Virus`
- `Tomato_Bacterial_spot`
- `Potato___Late_blight`
- `Pepper__bell___healthy`
- ... and more

## 🚀 How to Use
1. Run `test_single_image.py` to test your saved model on an image.
2. Place your image path inside the script.

## 📁 Files
- `train_cnn.py` – CNN training script
- `test_single_image.py` – Single image prediction script
- `model.pth` – Trained model (not included in repo)

## 🧾 Requirements
- PyTorch
- torchvision
- PIL

## 💡 Future Ideas
- Add GUI using Tkinter or Streamlit
- Integrate real-time webcam capture
- Use Git LFS to manage large `.pth` files

## 👤 Author
Ahasanur Rafi – North South University
