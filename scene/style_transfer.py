import torch
import torch.nn as nn
from scene.gaussian_conv import GaussianConv
from utils.loss_utils import calc_mean_std

class CNN(nn.Module):
    def __init__(self, matrixSize=32):
        super(CNN,self).__init__()
        # 256x64x64
        self.convs = nn.Sequential(nn.Conv2d(256,128,3,1,1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(128,64,3,1,1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(64,matrixSize,3,1,1))
        # 32x8x8
        self.fc = nn.Linear(matrixSize*matrixSize,matrixSize*matrixSize)
        #self.fc = nn.Linear(32*64,256*256)

    def forward(self,x):
        out = self.convs(x)
        # 32x8x8
        b,c,h,w = out.size()
        out = out.view(b,c,-1)
        # 32x64
        out = torch.bmm(out,out.transpose(1,2)).div(h*w)
        # 32x32
        out = out.view(out.size(0),-1)
        return self.fc(out)


class MulLayer(nn.Module):
    def __init__(self, matrixSize=32, adain=True):
        super(MulLayer,self).__init__()
        self.adain = adain
        if adain:
            return

        self.snet = CNN(matrixSize)
        self.matrixSize = matrixSize

        self.compress = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, matrixSize)
        )
        self.unzip = nn.Sequential(
            nn.Linear(matrixSize, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256)
        )


    def forward(self,cF,sF, trans=True):
        '''
        input:
            point cloud features: [N, C]
            style image features: [1, C, H, W]
            D: matrixSize
        '''
        if self.adain:
            cF = cF.T # [C, N]
            style_mean, style_std = calc_mean_std(sF) # [1, C, 1]
            content_mean, content_std = calc_mean_std(cF.unsqueeze(0)) # [1, C, 1]

            style_mean = style_mean.squeeze(0)
            style_std = style_std.squeeze(0)
            content_mean = content_mean.squeeze(0)
            content_std = content_std.squeeze(0)

            cF = (cF - content_mean) / content_std
            cF = cF * style_std + style_mean
            return cF.T
      
        assert cF.size(1) == sF.size(1), 'cF and sF must have the same channel size'
        assert sF.size(0) == 1, 'sF must have batch size 1'
        N, C = cF.size()
        B, C, H, W = sF.size()

        # normalize point cloud features
        cF = cF.T # [C, N]
        style_mean, style_std = calc_mean_std(sF) # [1, C, 1]
        content_mean, content_std = calc_mean_std(cF.unsqueeze(0)) # [1, C, 1]

        content_mean = content_mean.squeeze(0)
        content_std = content_std.squeeze(0)

        cF = (cF - content_mean) / content_std # [C, N]
        # compress point cloud features
        compress_content = self.compress(cF.T).T # [D, N]

        # normalize style image features
        sF = sF.view(B,C,-1)
        sF = (sF - style_mean) / style_std  # [1, C, H*W]

        if(trans):
            # get style transformation matrix
            sMatrix = self.snet(sF.reshape(B,C,H,W)) # [B=1, D*D]
            sMatrix = sMatrix.view(self.matrixSize,self.matrixSize) # [D, D]

            transfeature = torch.mm(sMatrix, compress_content).T # [N, D]
            out = self.unzip(transfeature).T # [C, N]

            style_mean = style_mean.squeeze(0) # [C, 1]
            style_std = style_std.squeeze(0) # [C, 1]

            out = out * style_std + style_mean
            return out.T # [N, C]
        else:
            out = self.unzip(compress_content.T) # [N, C]
            out = out * content_std + content_mean
            return out
