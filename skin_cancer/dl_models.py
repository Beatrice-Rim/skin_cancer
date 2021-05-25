from torch import nn
from torchvision import models


def turn_off_grad(model):
    for param in model.parameters():
        param.requires_grad = False


class PretrainedModel(nn.Module):
    def __init__(self, model, num_classes, pretrained=True):
        super(PretrainedModel, self).__init__()
        if model == 'alexnet':
            self.model = models.alexnet(pretrained=pretrained)
            if pretrained:
                turn_off_grad(self.model)
            num_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_features, num_classes)
        elif model == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
            if pretrained:
                turn_off_grad(self.model)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.model(x).squeeze()


class RegressionModel(nn.Module):
    def __init__(self, num_features, out_features):
        super(RegressionModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=num_features, out_features=out_features)
        )

    def forward(self, x):
        return self.layer(x).squeeze()


class MLP(nn.Module):
    def __init__(self, num_features, hid_features, out_features, dropout=0.15):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, hid_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_features, out_features)
        )

    def forward(self, x):
        return self.model(x).squeeze()
