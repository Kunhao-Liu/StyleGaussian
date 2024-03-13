import torch.nn as nn
from collections import namedtuple
import torchvision.models as models


import torchvision
normalize_vgg = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                 std=[0.229, 0.224, 0.225])

def denormalize_vgg(img):
    im = img.clone()
    im[:, 0, :, :] *= 0.229
    im[:, 1, :, :] *= 0.224
    im[:, 2, :, :] *= 0.225
    im[:, 0, :, :] += 0.485
    im[:, 1, :, :] += 0.456
    im[:, 2, :, :] += 0.406
    return im

# pytorch pretrained vgg
class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        #pretrained vgg19
        vgg19 = models.vgg19(weights='DEFAULT').features

        self.relu1_1 = vgg19[:2]
        self.relu2_1 = vgg19[2:7]
        self.relu3_1 = vgg19[7:12]
        self.relu4_1 = vgg19[12:21]

        #fix parameters
        self.requires_grad_(False)

    def forward(self, x):
        _output = namedtuple('output', ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'])
        relu1_1 = self.relu1_1(x)
        relu2_1 = self.relu2_1(relu1_1)
        relu3_1 = self.relu3_1(relu2_1)
        relu4_1 = self.relu4_1(relu3_1)
        output = _output(relu1_1, relu2_1, relu3_1, relu4_1)

        return output