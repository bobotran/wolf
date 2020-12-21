from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
import torch.nn as nn
import torch
import torch.nn.functional as F

class DigitResNet(ResNet):
    def __init__(self, embedding_dim=10):
        super(DigitResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10) # Based on ResNet18
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3,bias=False)
        # norm layer
        self.fc = nn.Linear(512, embedding_dim)

class ADMNet(nn.Module):
    def __init__(self, embedding_dim, num_classes=10):
        super(ADMNet, self).__init__() 
        self.base = DigitResNet(embedding_dim) # Outputs the embedding
        self.output = nn.Linear(embedding_dim, num_classes)
    def forward(self, x):
        x = self.base(x)
        x = F.normalize(x, dim=1)
        return x

class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim, num_classes=10):
        super(EmbeddingNet, self).__init__() 
        self.base = DigitResNet(embedding_dim) # Outputs the embedding
        self.output = nn.Sequential(
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_classes)
        )
    def forward(self, x):
        x = self.base(x)
        return x

class DigitFeatureExtractor(nn.Module):
    def __init__(self, dim, weights_path=None):
        super(DigitFeatureExtractor, self).__init__()

        self.net = ADMNet(dim) #TODO
        if weights_path is not None:
            self.net.load_state_dict(torch.load(weights_path))
            for param in self.net.parameters():
                param.requires_grad = False
            #self.net.eval()

    def forward(self, x):
        return self.net(x)
