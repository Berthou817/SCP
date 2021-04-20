import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import vgg19

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor,self).__init__()
        vgg19_model = vgg19(pretrained=True)
        vgg19_model_new = list(vgg19_model.features.children())[:18]
        vgg19_model_new[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.feature_extractor = nn.Sequential(*vgg19_model_new)
    def forward(self,img):
        return self.feature_extractor(img)

def pore(img):
    img = img.flatten()
    por = 0
    for i in range(len(img)):
        if img[i]==0: por +=1
    return por/len(img)

class My_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        mseloss = F.binary_cross_entropy(x,y)
        x = torch.where(x>0.5,torch.ones_like(x),torch.zeros_like(x))
        x = x.data.cpu().numpy().astype(np.int)
        y = y.data.cpu().numpy().astype(np.int)
        P = 0
        for i in range(20):
            por_pred = pore(x[i])
            pore_true = pore(y[i])
            por = (por_pred-pore_true)**2
            P = P+por

        PORE= P/20
        P=torch.tensor(PORE)
        return P+mseloss,PORE
