import torch
import torch.nn as nn
from torch.nn.modules.pooling import AdaptiveAvgPool2d
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class BottleNeck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(BottleNeck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x


class Classifier(nn.Module):
    def __init__(self, dim_bottleneck: int, num_classes: int, type: str = "ori"):
        super(Classifier, self).__init__()
        self.head = nn.Linear(dim_bottleneck, num_classes)
        if type == 'wn':
            self.head = weight_norm(self.head)
        self.head.apply(init_weights)

    def forward(self, x):
        return self.head(x)


# class ProtoClassifier(nn.Module):
#     def __init__(self, dim_bottleneck: int, num_classes: int, type: str = "ori", temp: float = 1.):
#         super().__init__()
#         self.head = nn.Linear(dim_bottleneck, num_classes)
#         self.temp = temp
#         if type == 'wn':
#             self.head = weight_norm(self.head)
#         self.head.apply(init_weights)

#     def forward(self, x):
#         return self.head(F.normalize(x))/self.temp
