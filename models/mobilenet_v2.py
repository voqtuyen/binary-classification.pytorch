import torch
from torch import nn
from torch import optim
from torchvision.models import mobilenet_v2


class mobilenet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        model_ft = mobilenet_v2(pretrained=cfg['model']['pretrained'])

        # Freeze network's weights, using ConvNet as fixed feature extractor
        for param in model_ft.parameters():
            param.requires_grad == False

        self.feat = model_ft.features
        self.classifier = nn.Linear(model_ft.classifier[1].in_features, cfg['model']['num_output'])

    def forward(self, x):
        x = self.feat(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)

        return x