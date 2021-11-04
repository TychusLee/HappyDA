import torchvision.models.resnet as resnet
import torch.nn as nn
import torch


class ResNet(nn.Module):
    def __init__(self, arch: str = 'resnet50', pretrained: bool = False, **kwargs):
        super().__init__()
        backbone = resnet.__dict__[arch](pretrained=pretrained, **kwargs)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.out_features = backbone.fc.in_features
        self.arch = arch
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = x.view(-1, self._out_features)
        return x
