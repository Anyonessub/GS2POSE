import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.models import vgg16_bn
import cv2 
from torchvision.models import vgg19

class VGGFeatureLoss(nn.Module):
    def __init__(self):
        super(VGGFeatureLoss, self).__init__()
        self.vgg = vgg19(pretrained=True).features.eval().cuda()
        self.layer_indices = [0, 5, 10, 19]  # Layers to use for feature extraction
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        input_features = self.get_features(input)
        target_features = self.get_features(target)

        loss = 0.0
        for input_feat, target_feat in zip(input_features, target_features):
            loss += F.mse_loss(input_feat, target_feat)

        return loss

    def get_features(self, x):
        features = []
        for i, layer in enumerate(self.vgg.children()):
            x = layer(x)
            if i in self.layer_indices:
                features.append(x)
        return features