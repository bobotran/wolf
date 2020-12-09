from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
import torch.nn as nn
import torch

class DigitResNet(ResNet):
    def __init__(self):
        super(DigitResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10) # Based on ResNet18
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3,bias=False)
        # norm layer

class DigitFeatureExtractor(nn.Module):
    def __init__(self, weights_path=None):
        super(DigitFeatureExtractor, self).__init__()

        self.net = DigitResNet()
        if weights_path is not None:
            self.net.load_state_dict(torch.load(weights_path))
            for param in self.net.parameters():
                param.requires_grad = False
            #self.net.eval()
        self.net.fc = nn.Flatten()

    def forward(self, x):
        return self.net(x)
