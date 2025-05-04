# Semantic-Segmentation-with-Deep-EncoderDecoder-Networks-A-SegNet-Approach

This project implements a semantic segmentation model based on the SegNet architecture to classify each pixel in road scene images into one of 11 classes (e.g., road, building, sky, pedestrian, etc.). It is designed for applications in autonomous driving and intelligent transportation systems.

---

## 📁 Project Structure
SegNetProject/
├── data/
│ ├── train/
│ ├── val/
│ └── test/
├── models/
│ └── segnet_model.py
├── results/
│ ├── predicted_masks/
│ └── evaluation_metrics.txt
├── utils/
│ └── dataloader.py
│ └── metrics.py
├── train.py
├── test.py

## 🚀 Key Features

- **Model**: Based on the [SegNet architecture](https://arxiv.org/abs/1511.00561) with encoder-decoder layers and max-pooling indices for upsampling.
- **Dataset**: Custom dataset for road scenes with 11 semantic classes.
- **Loss**: Cross-entropy loss with class weights (optional).
- **Metrics**: Pixel accuracy, Mean IoU, per-class IoU.
- **Visualization**: Saves predicted segmentation masks for test data.

🏗️ Model Architecture
The model uses:

Encoder: 5 convolution + max-pooling blocks (like VGG16)

Decoder: 5 upsampling + convolution blocks using max-pooling indices

Final Layer: Softmax over 11 classes


📊 Evaluation Metrics
Pixel Accuracy

Mean Intersection over Union (mIoU)

Per-Class Accuracy

👨‍💻 Author
Bhukya Dayanand
