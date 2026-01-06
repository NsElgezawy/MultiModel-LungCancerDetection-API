# Lung & Colon Cancer Classification

A Deep learning web application designed for classifying **Lung and Colon Cancer** histopathology. Leveraging ensemble architectures and Grad-CAM explainability, the system is trained on 25,000 images to robustly
identify five distinct tissue types:

**Lung**: Adenocarcinoma, Squamous Cell Carcinoma, and Benign Tissue.

**Colon**: Adenocarcinoma and Benign Tissue.

### Dataset  
- You can reach dataset through Kaggle via [Lung and Colon Cancer](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)


## Features
- **Multi-Model Support:** Choose between CNN, VGG16, ResNet50, ViT, or an Ensemble for predictions.
- **Explainability:** Visualizes model focus areas using **Grad-CAM** (Gradient-weighted Class Activation Mapping).
- **Out-of-Distribution (OOD) Detection:** Automatically flags images that are likely irrelevant or low-confidence to prevent false diagnoses.
- **Uncertainty Estimation:** Provides uncertainty metrics using MC Dropout.
- **Modern UI:** Clean, responsive interface with Dark Mode support.

---

## System Snapshots

![OncoAI_Homepage](https://cdn.jumpshare.com/preview/7fqxeCI3GqOWDR3Fa8mlmvpTnj1sWZVlM_z_aauYkvE6AGV22gtBov-RzuiqnSiLLfMbNuEWSnedCG1WKQg1FpFS7gV8buH7O8u79Biy4Qw) 

## Models Used

The application supports both custom and pretrained models to ensure robustness and accuracy:

### 1. Custom Models
- **DeepHistopathCNN:** A lightweight, custom-designed Convolutional Neural Network (CNN) optimized for histopathological texture analysis.

### 2. Pretrained Models (Transfer Learning)
- **VGG16:** A classic deep CNN architecture known for its simplicity and depth.
- **ResNet50:** A residual network that solves the vanishing gradient problem, allowing for deeper feature extraction.
- **Vision Transformer (ViT):** Applies self-attention mechanisms to image patches, capturing long-range dependencies in the data.

### 3. Ensemble
- **Soft Voting Ensemble:** Combines the predictions of **VIT,VGG16** to achieve higher accuracy and stability than any single model alone.

---

## Models Comparison
![Model_plot](https://cdn.jumpshare.com/preview/ebDW9o7OdEYjqnNR5GOrNn0AHqp3_UGR7l4PQjM2_D_34nCPhnH8y6s2GII7tPHJ1kjk6rZ_KZokmGTki2nG_2lNnvzSbGmqFIseaA31Lto)
## 🛠️ Installation & Local Run

Follow these steps to run the project locally on your machine.

### Prerequisites
- Python 3.8 or higher
- CUDA capable GPU (optional, but recommended for faster inference)

### 1. Clone the Repository
```bash
git clone https://github.com/NsElgezawy/MultiModel-LungCancerDetection-API
cd MultiModel-LungCancerDetection-API
```

### 2. Install Dependencies
It involves installing PyTorch, Flask, and other utilities.
```bash
pip install torch torchvision flask pillow opencv-python numpy
```

### 3. Organize Model Weights
Ensure you download Models .pth files from drive
[Model_dirve](https://drive.google.com/drive/folders/1Rat-uxRiNi63NqmzMFgClQsWDRGZY9YU?usp=sharing)


Trained model `.pth` files are placed in the `models/` directory:
- `models/cnn.pth`
- `models/vgg.pth`
- `models/resnet_feature_extraction_checkpoint.pth`
- `models/vit_feature_extraction_checkpoint.pth`

### 4. Run the Application
Start the Flask development server:
```bash
python app.py
```
Open your browser and navigate to: `http://127.0.0.1:5000`

---

## Project Structure
```
OncoAI/
├── app.py              # Main Flask application
├── inference.py        # Core logic for models, prediction, and Grad-CAM
├── requirements.txt    # List of python dependencies
├── static/             # CSS, JS, and images
├── templates/
│   └── index.html      # Frontend HTML template
├── models/             # Directory for .pth model weights
└── README.md           # Project documentation
```

---

## Project Contributors  

| Name            |
|-----------------|
| Ali Elbahrawy   |
| Anas Elgezawy   |
| Hagar Abo Samra |
| Hoda Mahmoud    |
| Mohamed Ahmed   |
| Mariam Ahmed    |
| Mohamed Ashraf  |
| Shahd Ahmed     |
