import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import vit_b_16
from PIL import Image
import os
import cv2
import numpy as np
import base64
from io import BytesIO
import gc  # Added for memory cleanup

# ==========================================
# Configuration & Paths
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
CNN_PATH = os.path.join(BASE_DIR, "models", "cnn.pth")
VGG_PATH = os.path.join(BASE_DIR, "models", "vgg.pth")
RESNET_PATH = os.path.join(BASE_DIR, "models", "resnet_feature_extraction_checkpoint.pth")
VIT_PATH = os.path.join(BASE_DIR, "models", "vit_feature_extraction_checkpoint.pth")

CLASS_NAMES = ["colon_aca", "colon_n", "lung_aca", "lung_n", "lung_scc"]

# ==========================================
# Out-of-Distribution Detection Thresholds
# ==========================================
CONFIDENCE_THRESHOLD = 60.0  # Minimum confidence % to accept prediction
UNCERTAINTY_THRESHOLD = 15.0  # Maximum uncertainty % to accept prediction

# Global cache to prevent reloading models from disk every request
MODEL_CACHE = {}

# ==========================================
# Transforms
# ==========================================
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ==========================================
# Model Definitions
# ==========================================
class DeepHistopathCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.4),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.4),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def get_vgg16_model(num_classes=5):
    vgg = models.vgg16(weights=None)
    vgg_features = vgg.features
    for param in vgg.parameters(): param.requires_grad = False
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

def get_resnet50_model(num_classes=5):
    model = models.resnet50(weights=None)
    for p in model.parameters(): p.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512), nn.ReLU(),
        nn.Dropout(0.3), nn.Linear(512, num_classes)
    )
    return model

def get_vit_model(num_classes=5):
    model = vit_b_16(weights=None)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model

class SoftVotingEnsemble(nn.Module):
    def __init__(self, models, weights=None):
        super(SoftVotingEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights if weights is not None else [1.0 / len(models)] * len(models)
        
    def forward(self, x):
        predictions = []
        for model in self.models:
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            predictions.append(probs)
        ensemble_probs = sum(w * p for w, p in zip(self.weights, predictions))
        return ensemble_probs

# ==========================================
# Caching & Preloading Logic (NEW)
# ==========================================
def load_model_once(model_name, device):
    """Loads model from disk only if not already in cache."""
    global MODEL_CACHE
    
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    print(f"⏳ Loading {model_name} from disk... (First run)")
    
    model = None
    
    if model_name == "cnn":
        model = DeepHistopathCNN(num_classes=5)
        model.load_state_dict(torch.load(CNN_PATH, map_location="cpu"))
        
    elif model_name == "vgg16":
        model = get_vgg16_model(num_classes=5)
        model.load_state_dict(torch.load(VGG_PATH, map_location="cpu"))
        
    elif model_name == "resnet":
        model = get_resnet50_model(num_classes=5)
        model.load_state_dict(torch.load(RESNET_PATH, map_location="cpu"))
        
    elif model_name == "vit":
        model = get_vit_model(num_classes=5)
        model.load_state_dict(torch.load(VIT_PATH, map_location="cpu"))
        
    elif model_name == "ensemble":
        vgg = load_model_once("vgg16", device)
        vit = load_model_once("vit", device)
        model = SoftVotingEnsemble([vgg, vit], weights=[0.5, 0.5])

    if model:
        model = model.to(device)
        model.eval()
        MODEL_CACHE[model_name] = model # Store in cache
        
    return model

def preload_all_models():
    """Call this on app startup to load everything into VRAM."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Preloading all models to {device}...")
    
    models_to_load = ["cnn", "vgg16", "resnet", "vit", "ensemble"]
    
    for name in models_to_load:
        load_model_once(name, device)
        
    print("✅ All models loaded and ready for inference.")

def free_memory():
    """Clears VRAM and Cache."""
    global MODEL_CACHE
    print("🧹 Clearing Model Cache and VRAM...")
    MODEL_CACHE.clear()
    gc.collect()
    torch.cuda.empty_cache()
    print("✅ Memory cleared.")

# ==========================================
# MC Dropout & Grad-CAM Logic
# ==========================================
def enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

def mc_dropout_predict_single(model, image_tensor, device, T=5, is_ensemble=False):
    # Optimized T=5 for speed (was 20)
    model.eval()
    enable_mc_dropout(model) 

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    image_tensor = image_tensor.to(device)

    probs_mc = []
    with torch.no_grad():
        for _ in range(T):
            outputs = model(image_tensor)
            if is_ensemble:
                probs = outputs
            else:
                probs = torch.softmax(outputs, dim=1)
            probs_mc.append(probs.cpu().numpy())

    probs_mc = np.array(probs_mc)
    mean_prob = probs_mc.mean(axis=0)
    std_prob  = probs_mc.std(axis=0)

    pred_idx = mean_prob.argmax()
    conf = mean_prob.max()
    uncertainty = std_prob[0][pred_idx]

    return pred_idx, conf, uncertainty

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Store hooks so we can remove them later
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))
    
    def generate(self, input_tensor, class_idx=None):
        # Enable gradients for backward pass
        self.model.eval()  # Keep batch norm in eval mode
        
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Get class index
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        elif isinstance(class_idx, torch.Tensor):
            class_idx = class_idx.item()
        
        # Backward pass for the target class
        score = output[0, class_idx]
        score.backward()
        
        # Generate CAM
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured. Check hooks.")
        
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize CAM with safety checks
        cam_min = cam.min()
        cam_max = cam.max()
        
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            # If CAM is uniform, return zeros
            cam = torch.zeros_like(cam)
        
        return cam.squeeze()
    
    def remove_hooks(self):
        """Clean up hooks to prevent memory leaks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def compute_gradcam(images):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Use cached ResNet
        model = load_model_once("resnet", device)
        
        target_layer = model.layer4[-1]
        gradcam = GradCAM(model, target_layer)
        
        # Convert image to RGB if needed
        img = images[0]
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize for display
        original_pil = img.resize((224, 224))
        
        # Prepare tensor for model (with gradients enabled)
        input_tensor = val_test_transform(img).unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)
        
        # Generate CAM
        cam = gradcam.generate(input_tensor)
        
        # Clean up hooks
        gradcam.remove_hooks()
        
        # Convert to numpy for visualization
        img_np = np.array(original_pil).astype(np.float32) / 255.0
        cam_np = cam.cpu().detach().numpy()
        
        # Ensure CAM is 2D
        if cam_np.ndim > 2:
            cam_np = cam_np.squeeze()
        
        # Resize CAM to match image
        cam_resized = cv2.resize(cam_np, (224, 224))
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Create overlay (adjust alpha for better visibility)
        overlay = 0.6 * img_np + 0.4 * heatmap
        overlay = np.clip(overlay, 0, 1)
        overlay_uint8 = np.uint8(255 * overlay)
        
        # Convert to base64
        pil_img = Image.fromarray(overlay_uint8)
        buff = BytesIO()
        pil_img.save(buff, format="PNG")
        return base64.b64encode(buff.getvalue()).decode("utf-8")
        
    except Exception as e:
        # Return error message instead of crashing
        print(f"Grad-CAM Error: {str(e)}")
        raise RuntimeError(f"Grad-CAM generation failed: {str(e)}")


# ==========================================
# Main Predict Function
# ==========================================
def predict(model_name, images):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Safety check for list
    if not isinstance(images, list):
        images = [images]

    if model_name == "gradcam":
        try:
            return compute_gradcam(images), 100.0, 0.0
        except Exception as e:
            return str(e), 0, 0

    # Load Model (Checks cache first)
    try:
        model = load_model_once(model_name, device)
    except Exception as e:
        return f"Error loading model: {str(e)}", 0, 0

    is_ensemble = (model_name == "ensemble")
    img_tensor = val_test_transform(images[0])

    # Run Prediction
    # Note: T=5 is used for faster response time. Increase if more precision is needed.
    pred_idx, conf, unc = mc_dropout_predict_single(
        model, img_tensor, device, T=10, is_ensemble=is_ensemble
    )

    prediction = CLASS_NAMES[pred_idx]
    confidence = round(conf.item() * 100, 2)
    uncertainty = round(unc.item() * 100, 2)

    # ==========================================
    # Out-of-Distribution Detection
    # ==========================================
    # If confidence is too low OR uncertainty is too high,
    # the image is likely not a histopathology sample
    if confidence < CONFIDENCE_THRESHOLD or uncertainty > UNCERTAINTY_THRESHOLD:
        prediction = "unknown"
        # Note: We still return the confidence and uncertainty for analysis

    return prediction, confidence, uncertainty