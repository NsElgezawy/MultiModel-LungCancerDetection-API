import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import vit_b_16
from PIL import Image



import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # ديركتوري الحالي
CNN_PATH = os.path.join(BASE_DIR, "models", "cnn.pth")
VGG_PATH = os.path.join(BASE_DIR, "models", "vgg.pth")
RESNET_PATH = os.path.join(BASE_DIR, "models", "resnet_feature_extraction_checkpoint.pth")
VIT_PATH = os.path.join(BASE_DIR, "models", "vit_feature_extraction_checkpoint.pth")


# ======= Transform =======
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

CLASS_NAMES = ["colon_aca", "colon_n", "lung_aca", "lung_n", "lung_scc"]

# ======= CNN =======
class DeepHistopathCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 112x112
            nn.Dropout(0.2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 56x56
            nn.Dropout(0.3),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 28x28
            nn.Dropout(0.4),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 14x14
            nn.Dropout(0.4),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ======= VGG16 =======
def get_vgg16_model(num_classes=5):
    vgg = models.vgg16(weights=None)
    vgg_features = vgg.features
    for param in vgg.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        nn.Linear(512*7*7, 1024), nn.ReLU(),
        nn.Linear(1024, 512), nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

    class VGG_Torch(nn.Module):
        def __init__(self, features, classifier):
            super().__init__()
            self.features = features
            self.avgpool = nn.AdaptiveAvgPool2d((7,7))
            self.classifier = classifier
        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x,1)
            x = self.classifier(x)
            return x

    return VGG_Torch(vgg_features, classifier)

# ======= ResNet50 =======
def get_resnet50_model(num_classes=5):
    model = models.resnet50(weights=None)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model

# ======= ViT =======
def get_vit_model(num_classes=5):
    model = vit_b_16(weights=None)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model

# ======= Predict =======
def predict(model_name, images):
    device = torch.device("cpu")

    if model_name=="cnn":
        model = DeepHistopathCNN(num_classes=5)
        model.load_state_dict(torch.load(CNN_PATH, map_location=device))
    elif model_name=="vgg16":
        model = get_vgg16_model(num_classes=5)
        model.load_state_dict(torch.load(VGG_PATH, map_location=device))
    elif model_name=="resnet":
        model = get_resnet50_model(num_classes=5)
        model.load_state_dict(torch.load(RESNET_PATH, map_location=device))
    elif model_name=="vit":
        model = get_vit_model(num_classes=5)
        model.load_state_dict(torch.load(VIT_PATH, map_location=device))
    else:
        return "Unknown model", 0

    model.eval()
    img = val_test_transform(images[0]).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probs = F.softmax(output, dim=1)
        conf, pred = torch.max(probs,1)

    prediction = CLASS_NAMES[pred.item()]
    confidence = round(conf.item()*100,2)
    return prediction, confidence
