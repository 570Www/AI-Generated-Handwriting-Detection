import os
import json
import torch
import argparse
from torchvision import transforms
from PIL import Image
from discriminator import *
import numpy as np

def load_ddp_checkpoint(model, ckpt_path, map_location="cpu"):
    state_dict = torch.load(ckpt_path, map_location=map_location)

    # Remove 'module.' prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[len("module."):]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    return model

# =========================================================
# Inference on Single Image
# =========================================================
def infer_single_image(model, image_path, device):
    model.eval()
    
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((64, 128)),
        transforms.ToTensor(),
    ])
    
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Forward pass
    with torch.no_grad():
        logits = model(img)
    
    # Apply softmax to get class probabilities
    probs = torch.softmax(logits, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()

    # Return probabilities and predicted class
    return probs.cpu().numpy(), pred_class

# =========================================================
# Main for Inference
# =========================================================
def main():
    parser = argparse.ArgumentParser("Evaluate Handwriting Classifier")

    # -------------------------
    # Runtime
    # -------------------------
    parser.add_argument("--image_path", type=str, default="/home/againsturb/570/code/AIHandDis/experiments/data/Uni_ai_perturbed/test/DiffBrush_00013.png")
    parser.add_argument("--ckpt", type=str, default="./checkpoints/B/Uni_ai/discriminator_epoch_5.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # Model
    # -------------------------
    model = DeiTDualHandwritingClassifier().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["discriminator"])

    model.eval()

    # -------------------------
    # Inference
    # -------------------------
    probs, pred_class = infer_single_image(model, args.image_path, device)

    print(f"Human:", probs[0, 0])
    print(f"AI:", probs[0, 1])
    print(f"Predict_class:", "AI" if pred_class==1 else "Human")


if __name__ == "__main__":
    main()
