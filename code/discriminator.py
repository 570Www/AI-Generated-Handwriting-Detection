import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import timm


class DeiTDualHandwritingClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # --------------------------
        # DeiT encoder + processor
        # --------------------------
        self.backbone = timm.create_model(
            "deit_small_patch16_224",
            pretrained=True,
            num_classes=0  # Removes the final classification head
        )
        self.DeiT_feat_dim = self.backbone.num_features
        self.backbone.requires_grad_(False)

        self.load_size = 384

        # --------------------------
        # CNN branch (ResNet-18)
        # --------------------------
        shufflenet = models.shufflenet_v2_x1_0(weights="IMAGENET1K_V1")
        shufflenet.fc = nn.Identity()  # remove classification head
        self.cnn = shufflenet
        self.cnn_feat_dim = 1024

        self.feat4 = self.feat5 = self.feat6 = None

        def _hook(idx):
            def hook(m, i, o):
                feat = (
                    o[:, 1:]
                    .transpose(1, 2)
                    .contiguous()
                    .view(o.size(0), 384, 14, 14)
                )
                setattr(self, f"feat{idx}", feat)
            return hook

        self.backbone.blocks[4].register_forward_hook(_hook(4))
        self.backbone.blocks[5].register_forward_hook(_hook(5))
        self.backbone.blocks[6].register_forward_hook(_hook(6))

        # --------------------------
        # Fusion classifier
        # --------------------------
        self.classifier = nn.Sequential(
            nn.Linear(self.DeiT_feat_dim + self.cnn_feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 2)
        )

    def forward(self, images):
        """
        images: (B, 3, 64, 128)
        """

        # --------------------------------------
        # CNN BRANCH (raw images 64×128)
        # --------------------------------------
        with torch.no_grad():
            visual_feat = self.cnn(images)  # (B, 512)

        # --------------------------------------
        # CLIP BRANCH
        # Must run preprocessing!
        # --------------------------------------
        # with torch.no_grad():
        DeiT_outputs = self.backbone(F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False))

        # --------------------------------------
        # Fusion
        # --------------------------------------
        fused = torch.cat([visual_feat, DeiT_outputs], dim=-1)  # (B, 1152+512)

        # Classify
        logits = self.classifier(fused)

        return logits
    
class TSDHandwritingClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # --------------------------
        # DeiT encoder + processor
        # --------------------------
        self.backbone = timm.create_model(
            "deit_small_patch16_224",
            pretrained=True,
            num_classes=0  # Removes the final classification head
        )
        self.DeiT_feat_dim = self.backbone.num_features
        self.backbone.requires_grad_(False)

        self.load_size = 384

        # --------------------------
        # CNN branch (ResNet-18)
        # --------------------------
        shufflenet = models.shufflenet_v2_x1_0(weights="IMAGENET1K_V1")
        shufflenet.fc = nn.Identity()  # remove classification head
        self.cnn = shufflenet
        self.cnn_feat_dim = 1024

        # --------------------------
        # Fusion classifier
        # --------------------------
        self.classifier = nn.Sequential(
            nn.Linear(self.DeiT_feat_dim + self.cnn_feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 2)
        )

    def forward(self, images):
        """
        images: (B, 3, 64, 128)
        """

        # --------------------------------------
        # CNN BRANCH (raw images 64×128)
        # --------------------------------------
        with torch.no_grad():
            visual_feat = self.cnn(images)  # (B, 512)

        # --------------------------------------
        # CLIP BRANCH
        # Must run preprocessing!
        # --------------------------------------
        # with torch.no_grad():
        DeiT_outputs = self.backbone(F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False))

        # --------------------------------------
        # Fusion
        # --------------------------------------
        fused = torch.cat([visual_feat, DeiT_outputs], dim=-1)  # (B, 1152+512)

        # Classify
        logits = self.classifier(fused)

        return logits

class ResNet18HandwritingClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        cnn = models.resnet18(weights="IMAGENET1K_V1")
        cnn.fc = nn.Identity()
        self.cnn = cnn
        self.cnn_feat_dim = 512

        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 2)
        )

    def forward(self, images):
        visual_feat = self.cnn(images)  # (B, 512)
        logits = self.classifier(visual_feat)  # (B, 2)

        return logits


class ResNet34HandwritingClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        cnn = models.resnet34(weights="IMAGENET1K_V1")
        cnn.fc = nn.Identity()
        self.cnn = cnn
        self.cnn_feat_dim = 512

        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 2)
        )

    def forward(self, images):
        visual_feat = self.cnn(images)
        logits = self.classifier(visual_feat)
        return logits


class ResNet50HandwritingClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        cnn = models.resnet50(weights="IMAGENET1K_V1")
        cnn.fc = nn.Identity()
        self.cnn = cnn
        self.cnn_feat_dim = 2048

        self.classifier = nn.Sequential(
            nn.Linear(self.cnn_feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 2)
        )

    def forward(self, images):
        visual_feat = self.cnn(images)
        logits = self.classifier(visual_feat)
        return logits




class ShuffleNetV2_Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        shufflenet = models.shufflenet_v2_x1_0(weights="IMAGENET1K_V1")
        shufflenet.fc = nn.Identity()
        self.cnn = shufflenet
        self.feat_dim = 1024

        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 2)
        )

    def forward(self, images):
        feat = self.cnn(images)
        logits = self.classifier(feat)
        return logits
    
class EfficientNetB0_Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        # --------------------------
        # Backbone: EfficientNetB0
        # --------------------------
        efficientnet = models.efficientnet_b0(weights="IMAGENET1K_V1")
        efficientnet.classifier = nn.Identity()  # remove classification head
        self.cnn = efficientnet
        self.feat_dim = 1280  # EfficientNetB0 feature dimension

        # --------------------------
        # Classifier
        # --------------------------
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 2)  # binary classification
        )

    def forward(self, images):
        feat = self.cnn(images)  # (B, 1280)
        logits = self.classifier(feat)
        return logits


class DeiT_Small_Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "deit_tiny_patch16_224",
            pretrained=True,
            num_classes=0
        )

        self.feat_dim = self.backbone.num_features  # 384 (DeiT-Small output dimension)

        # self.feat4 = self.feat5 = self.feat6 = None

        # self.backbone.blocks[3].register_forward_hook(
        #     lambda m,i,o: setattr(self, "feat4", o[:,1:].transpose(1,2).view(-1,384,14,14))
        # )
        # self.backbone.blocks[4].register_forward_hook(
        #     lambda m,i,o: setattr(self, "feat5", o[:,1:].transpose(1,2).view(-1,384,14,14))
        # )
        # self.backbone.blocks[5].register_forward_hook(
        #     lambda m,i,o: setattr(self, "feat6", o[:,1:].transpose(1,2).view(-1,384,14,14))
        # )

        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 2)
        )

    def forward(self, images):
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        feat = self.backbone(images)  # (B, 384)
        logits = self.classifier(feat)
        return logits
