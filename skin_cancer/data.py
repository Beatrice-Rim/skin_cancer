from pathlib import Path

import torchvision.transforms.functional as TF
from numpy import random
from torchvision import transforms
from torchvision.datasets import ImageFolder

degree = 30
translate = (0.1, 0.1)
scale = (0.8, 1.2)
gamma = (0.5, 1.5)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


class RandomGammaCorrection:
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, x):
        g = random.uniform(self.low, self.high)
        return TF.adjust_gamma(x, g)


transforms = {
    'train': transforms.Compose([
        transforms.RandomAffine(degree, translate, scale),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        RandomGammaCorrection(gamma[0], gamma[1]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
}


class SkinDataset(ImageFolder):
    def __init__(self, root, split):
        super(SkinDataset, self).__init__(Path(root)/split, transforms[split])