# 📁 model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class HybridFeatureExtractor(nn.Module):
    def __init__(self):
        super(HybridFeatureExtractor, self).__init__()
        self.effnet = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)

    def forward(self, x):
        eff_feats = F.normalize(self.effnet(x), p=2, dim=1)
        vit_feats = F.normalize(self.vit(x), p=2, dim=1)
        return torch.cat([eff_feats, vit_feats], dim=1)