#%% python packages
import torch
import torch.nn as nn
from torch.nn import init

try:
    from src.common_blocks import *
    from config.model_params import *
    from utils.img_utils import *
except:
    from ucc_13Spectral.src.common_blocks import *
    from ucc_13Spectral.config.model_params import *
    from ucc_13Spectral.utils.img_utils import *

__all__ = ['UCC', 'custom_loss']


def custom_loss(probability_map: torch.Tensor, error_map: torch.Tensor):
    loss = probability_map * error_map
    loss = loss.sum()
    return loss
#%% ucc_13spectral

class UNet(nn.Module):

    def __init__(self, in_channels=2, out_channels=1):
        super().__init__()
        self.layer1 = Conv(in_channels, in_channels, 
                           3, 1, norm='bn')
        self.layer2 = nn.Sequential(nn.AvgPool2d(2),
                                    Conv(in_channels, 4, 3, 1))
        self.layer3 = nn.Sequential(nn.AvgPool2d(2), Conv(4,
                                                          8,
                                                          3,
                                                          1,
                                                          norm='bn'))
        self.layer4 = nn.Sequential(nn.AvgPool2d(2),
                                    Conv(8, 16, 3, 1, norm='bn'))
        self.layer5 = nn.Sequential(nn.AvgPool2d(2),
                                    Conv(16, 16, 3, 1, norm='bn'))
        
        # for 13-channel spectral information
        # deconvolution the introduced spectral information
        self.upconv0 = nn.Sequential(
            nn.ConvTranspose2d(13,13,2,2),
            nn.BatchNorm2d(13),
            nn.ReLU(True),
            nn.ConvTranspose2d(13,13,2,2),
            nn.BatchNorm2d(13),
            nn.ReLU(True)
        )
        # convolution the feature fusing original and new feature map
        self.uplayer0 = Conv(29, 16, 3, 1, norm='bn')
        # --------DONE----------

        self.upconv5 = nn.ConvTranspose2d(16, 16, 2, 2)
        self.uplayer4 = Conv(32, 16, 3, 1, norm='bn')

        self.upconv4 = nn.ConvTranspose2d(16, 8, 2, 2)
        self.uplayer3 = Conv(16, 8, 3, 1, norm='bn')

        self.upconv3 = nn.ConvTranspose2d(8, 4, 2, 2)
        self.uplayer2 = Conv(8, 4, 3, 1, norm='bn')

        self.upconv2 = nn.ConvTranspose2d(4, 2, 2, 2)
        self.uplayer1 = Conv(in_channels + 2, in_channels, 3, 1, norm='bn')

        self.final_out = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, y=None):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)

        # x5 the initial feature map
        if y != None:
            y = self.upconv0(y)
            x5 = torch.cat([x5,y],dim=1)
            x5 = self.uplayer0(x5)

        x = self.upconv5(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.uplayer4(x)

        x = self.upconv4(x)
        x = torch.cat([x, x3], dim=1)
        x = self.uplayer3(x)

        x = self.upconv3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.uplayer2(x)

        x = self.upconv2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.uplayer1(x)

        x = self.final_out(x)
        return x

# %%
class UCC(nn.Module):

    def __init__(self, in_channels=1, out_channels=1):
        super(UCC, self).__init__()
        u_coord, v_coord = get_uv_coord(CustomParams.bin_num,
                                        range=CustomParams.boundary_value *
                                        2)
        uv_coords = torch.stack([u_coord, v_coord], dim=-1)
        self.rgb_map = log_uv_to_rgb_torch(uv_coords)  # h * w * 3

        self.rgb_map = self.rgb_map.reshape(-1, 3)
        if CustomParams.edge_info:
            in_channels += 1
        if CustomParams.coords_map:
            in_channels += 2
        self.model = UNet(in_channels, out_channels)
        self.init_params()

    def forward(self, x, y=None):
        h, w = x.shape[-2:]
        # x: img_uv, y: spectral info
        x = self.model(x, y) 
        x = x.reshape(x.shape[0], -1)
        probability_map = F.softmax(x, dim=-1)
        probability_map = probability_map.reshape(-1, h, w)
        return probability_map

    def inference(self, x, y=None):
        self.rgb_map = self.rgb_map.to(x.device)
        # x: img_uv, y: spectral info
        probability_map = self.forward(x, y) 
        b = probability_map.shape[0] # batch size
        probability_map = probability_map.reshape(b, -1)
        max_indexes = torch.argmax(probability_map, dim=-1)
        ret = self.rgb_map[max_indexes]
        return ret  # b * 3

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
