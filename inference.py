import torch

def predict(model_name, images):

    if model_name == "vgg16":
        model = torch.load("models/vgg.pth", map_location="cpu")

    elif model_name == "cnn":
        model = torch.load("models/cnn.pth", map_location="cpu")

    model.eval()

    # هنا عادة preprocessing + inference
    return "Cancer Detected"
