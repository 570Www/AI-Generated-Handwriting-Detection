import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelProj(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.proj(x)

class ConvBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.block(x)

class CrossBlockAttention(nn.Module):
    """
    Q = block5
    K,V = concat(block4, block6)
    Operates on per-block embed_ch channels
    """
    def __init__(self, ch):
        super().__init__()
        self.scale = ch ** -0.5

        self.q_proj = nn.Conv2d(ch, ch, 1)
        self.k_proj = nn.Conv2d(ch * 2, ch, 1)
        self.v_proj = nn.Conv2d(ch * 2, ch, 1)

        self.out = nn.Conv2d(ch, ch, 1)

    def forward(self, f4, f5, f6):
        """
        f4,f5,f6: (B, ch, 14, 14)
        """
        B, C, H, W = f5.shape

        q = self.q_proj(f5).view(B, C, -1)             # (B,C,196)

        kv = torch.cat([f4, f6], dim=1)                # (B,2C,14,14)
        k = self.k_proj(kv).view(B, C, -1)             # (B,C,392)
        v = self.v_proj(kv).view(B, C, -1)             # (B,C,392)

        attn = torch.softmax(
            torch.bmm(q.transpose(1, 2), k) * self.scale,
            dim=-1
        )                                              # (B,196,392)

        out = torch.bmm(attn, v.transpose(1, 2))       # (B,196,C)
        out = out.transpose(1, 2).view(B, C, H, W)

        return self.out(out) + f5


class LowRankDeformation(nn.Module):
    def __init__(self, ch, K=8):
        super().__init__()
        self.K = K

        # Basis generator (spatial)
        self.basis_net = nn.Sequential(
            ConvBlock(ch),
            nn.Conv2d(ch, K, kernel_size=1)
        )

        # Coefficient predictor (global semantic)
        self.coeff_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, ch // 2),
            nn.ReLU(inplace=False),
            nn.Linear(ch // 2, K)
        )

    def forward(self, x):
        """
        x: (B, C, 14, 14)
        """
        B, C, H, W = x.shape

        Bk = self.basis_net(x)                  # (B,K,14,14)
        alpha = self.coeff_net(x)               # (B,K)
        alpha = alpha.view(B, self.K, 1, 1)

        delta_patch = (Bk * alpha).sum(dim=1, keepdim=True)
        return delta_patch                      # (B,1,14,14)

class BADNet(nn.Module):
    """
    Boundary Aware Deformation Network
    """

    def __init__(self,
                 in_ch=384,
                 embed_ch=128,
                 K=8,
                 eps=2/255):
        super().__init__()

        self.eps = eps

        # ---- Channel alignment for blocks 4,5,6 ----
        self.proj4 = ChannelProj(in_ch, embed_ch)
        self.proj5 = ChannelProj(in_ch, embed_ch)
        self.proj6 = ChannelProj(in_ch, embed_ch)

        self.merge = nn.Conv2d(embed_ch * 3, embed_ch * 3, 1)

        # ---- Cross-block semantic attention ----
        self.attn = CrossBlockAttention(embed_ch)

        # ---- Semantic refinement ----
        self.refine = ConvBlock(embed_ch * 3)

        # ---- Low-rank deformation bottleneck ----
        self.lowrank = LowRankDeformation(embed_ch * 3, K=K)

        # ---- Image-space decoder ----
        self.decoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(16, 3, 3, padding=1)
        )

    def forward(self, x, feat4, feat5, feat6):
        """
        x     : (B,3,H,W)
        feat4 : (B,384,14,14)
        feat5 : (B,384,14,14)
        feat6 : (B,384,14,14)
        """

        # ---- Project channels ----
        f4 = self.proj4(feat4)
        f5 = self.proj5(feat5)
        f6 = self.proj6(feat6)

        f5_attn = self.attn(f4, f5, f6)   # (B,128,14,14)
        f = torch.cat([f4, f5_attn, f6], dim=1)   # (B,384,14,14)
        f = self.merge(f)

        f = self.refine(f)

        # ---- Low-rank deformation in patch space ----
        delta_patch = self.lowrank(f)    # (B,1,14,14)

        # ---- Patch-aligned upsampling (ViT-consistent) ----
        delta_img = F.interpolate(
            delta_patch,
            size=x.shape[-2:],
            mode="nearest"
        )

        delta_img = self.decoder(delta_img)
        delta_img = self.eps * torch.tanh(delta_img)

        x_hard = torch.clamp(x + delta_img, 0.0, 1.0)

        return x_hard, delta_img
