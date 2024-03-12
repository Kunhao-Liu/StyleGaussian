import torch

def positional_encoding(positions, freqs):
    
        freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
        pts = (positions[..., None] * freq_bands).reshape(
            positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
        pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
        return pts


class LinearLayer(torch.nn.Module):
    def __init__(self, inChanel, feape=0, out_dim=256):
        super(LinearLayer, self).__init__()

        self.in_mlpC = 2*feape*inChanel + inChanel
        self.feape = feape
        self.layer = torch.nn.Linear(self.in_mlpC, out_dim)

    def forward(self, features, sum_w):
        '''
        Args:
            features: the volumn-rendered features: [D, H, W]
            sum_w: the sum of weights along a ray: [H, W]
        '''
        D, H, W = features.shape
        features = features.permute(1,2,0).reshape(-1, D) # [H*W, D]
        sum_w = sum_w.reshape(-1, 1) # [H*W, 1]

        indata = [features]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        mlp_in = torch.cat(indata, dim=-1)

        out = mlp_in @ self.layer.weight.T + sum_w * self.layer.bias[None,...] # [H*W, out_dim]
        
        return out.reshape(H, W, -1).permute(2,0,1) # [out_dim, H, W]
    
    @torch.no_grad()
    def forward_directly_on_point(self, point_features):
        '''
        Args:
            features: features of the points: [N, D]
        '''
        indata = [point_features]
        if self.feape > 0:
            indata += [positional_encoding(point_features, self.feape)]
        mlp_in = torch.cat(indata, dim=-1)

        return self.layer(mlp_in) # [N, out_dim]
