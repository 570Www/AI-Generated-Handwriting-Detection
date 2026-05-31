import os
import glob
import io
from PIL import Image
import cv2
import numpy as np
import torch
import random
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torchvision.transforms.functional import resize, pad

def compute_laplacian(img_tensor):
    """
    img_tensor: (3, H, W), float32 in [0,1]
    return:     (1, H, W), zero-mean, unit-std per image
    """

    # 1. grayscale (keep channel dim)
    gray = img_tensor.mean(dim=0, keepdim=True)  # (1,H,W)

    # 2. Laplacian kernel (fixed)
    lap = (
        4 * gray[:, 1:-1, 1:-1]
        - gray[:, :-2, 1:-1]
        - gray[:, 2:, 1:-1]
        - gray[:, 1:-1, :-2]
        - gray[:, 1:-1, 2:]
    )

    # 3. pad back to original size
    lap = F.pad(lap, (1, 1, 1, 1))

    # 4. 🔑 per-image normalization (THIS IS THE KEY)
    lap = lap - lap.mean()
    lap = lap / (lap.std() + 1e-6)

    return lap

class PadToWidth:
    def __init__(self, target_width=1024):
        self.target_width = target_width

    def __call__(self, img):
        w, h = img.size

        # Resize height to 64, preserve aspect ratio
        new_h = 64
        new_w = int(w * (new_h / h))
        img = resize(img, (new_h, new_w))

        # Pad on the right
        if new_w < self.target_width:
            img = pad(img, (0, 0, self.target_width - new_w, 0), fill=255)
        else:
            # Clip if too long
            img = resize(img, (new_h, self.target_width))

        return img


class HandwritingDataset_word(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.cnn_size = (64, 128)
        self.transform = transforms.Compose([
            transforms.Resize(self.cnn_size),
            transforms.ToTensor()
        ])
        
    @staticmethod
    def from_root(root, train_folder="train", val_folder="val", test_folder='test', data=None):
        train_samples = []
        val_samples = []
        test_samples = []

        for label, folder in [(0, "human_word"), (1, data)]:
            for f in glob.glob(os.path.join(root, folder, train_folder, "*.png")):
                train_samples.append((f, label))

            for f in glob.glob(os.path.join(root, folder, val_folder, "*.png")):
                val_samples.append((f, label))

            for f in glob.glob(os.path.join(root, folder, test_folder, "*.png")):
                test_samples.append((f, label))

        train_dataset = HandwritingDataset_word(train_samples)
        val_dataset = HandwritingDataset_word(val_samples)
        test_dataset = HandwritingDataset_word(test_samples)
        return train_dataset, val_dataset, test_dataset

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label

class HandwritingDataset_word_Uni(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.cnn_size = (64, 128)
        self.transform = transforms.Compose([
            transforms.Resize(self.cnn_size),
            transforms.ToTensor()
        ])
        
    @staticmethod
    def from_root(root, train_folder="train", val_folder="val", test_folder='test', data=None):
        train_samples = []
        val_samples = []
        test_samples = []

        for label, folder in [(0, "Uni_human"), (1, data)]:
            for f in glob.glob(os.path.join(root, folder, train_folder, "*.png")):
                train_samples.append((f, label))

            for f in glob.glob(os.path.join(root, folder, val_folder, "*.png")):
                val_samples.append((f, label))

            for f in glob.glob(os.path.join(root, folder, test_folder, "*.png")):
                test_samples.append((f, label))

        train_dataset = HandwritingDataset_word(train_samples)
        val_dataset = HandwritingDataset_word(val_samples)
        test_dataset = HandwritingDataset_word(test_samples)
        return train_dataset, val_dataset, test_dataset

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label

""" class HandwritingDataset_word(Dataset):
    def __init__(self, samples, jpeg_quality=None, jpeg_prob=1.0):
        self.samples = samples
        self.cnn_size = (64, 128)

        # added (default = disabled)
        self.jpeg_quality = jpeg_quality
        self.jpeg_prob = jpeg_prob

        self.transform = transforms.Compose([
            transforms.Resize(self.cnn_size),
            transforms.ToTensor()
        ])

    @staticmethod
    def from_root(root, train_folder="train", val_folder="val", data=None,
                  jpeg_quality=None, jpeg_prob=1.0):
        train_samples = []
        val_samples = []

        for label, folder in [(0, "Uni_human"), (1, data)]:
            for f in glob.glob(os.path.join(root, folder, train_folder, "*.png")):
                train_samples.append((f, label))

            for f in glob.glob(os.path.join(root, folder, val_folder, "*.png")):
                val_samples.append((f, label))

        # pass through unchanged
        train_dataset = HandwritingDataset_word(
            train_samples, jpeg_quality=None, jpeg_prob=jpeg_prob
        )
        val_dataset = HandwritingDataset_word(
            val_samples, jpeg_quality=None, jpeg_prob=jpeg_prob
        )
        return train_dataset, val_dataset

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        img = Image.open(path).convert("RGB")

        # =====================================================
        # minimal JPEG compression injection
        # =====================================================
        if self.jpeg_quality is not None and random.random() < self.jpeg_prob:
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=self.jpeg_quality)
            buffer.seek(0)
            img = Image.open(buffer).convert("RGB")

        img = self.transform(img)
        img_lap = compute_laplacian(img)
        return img, img_lap, label """