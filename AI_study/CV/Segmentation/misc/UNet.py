from torchvision.transforms import CenterCrop
from torch.nn import ConvTranspose2d
from torch.nn import functional as F
from torch.nn import ModuleList
from torch.nn import MaxPool2d
from torch.nn import Conv2d
from torch.nn import Module
from torch.nn import ReLU
import torch

from misc import config


## Encoder, Decoder에 사용될 Down sample Block
class DownBlock(Module):

    def __init__(self, in_channel, out_channel):

        super(DownBlock, self).__init__()
        self.conv1 = Conv2d(in_channel, out_channel, 3)
        self.relu  = ReLU()
        self.conv2 = Conv2d(out_channel, out_channel, 3)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


## Decoder에 사용될 Up sample Block
class UpBlock(Module):

    def __init__(self, in_channel, out_channel):

        super(UpBlock, self).__init__()
        self.ConvT = ConvTranspose2d(in_channel, out_channel, 2, 2)


    def forward(self, x):

        return self.ConvT(x)


## U-Net Encoder
class Encoder(Module):

    def __init__(self, channels = (3, 16, 32, 64)):

        super(Encoder, self).__init__()

        ## 3 chn -> 16 chn * 2 -> 32 chn * 2 -> 64 chn * 2
        #! *2 : 앞에 있는 채널로 두 번
        self.enc_blocks = ModuleList(
            [DownBlock(channels[idx], channels[idx + 1])
                for idx in range(len(channels) - 1)])
        
        self.pool       = MaxPool2d(2)

    
    def forward(self, x):

        block_outputs = []
        for block in self.enc_blocks:

            x = block(x)
            ## Decoder랑 붙을 녀석들
            block_outputs.append(x)
            x = self.pool(x)

        return block_outputs


## U-Net Decoder
class Decoder(Module):

    def __init__(self, channels = (64, 32, 16)):

        super(Decoder, self).__init__()
        self.channels  = channels

        ## 64 chn -> 32 chn -> 16 chn
        self.up_convs  = ModuleList([
            UpBlock(channels[idx], channels[idx + 1])
            for idx in range(len(channels) - 1)
        ])

        self.dec_blocks = ModuleList([
            DownBlock(channels[idx], channels[idx + 1])
            for idx in range(len(channels) - 1)
        ])


    ## encoder output을 decoder의 up block에서
    ## 나온 feature 사이즈와 동일하게 center crop
    def crop(self, enc_feats, x):

        (_, _, H, W) = x.shape
        enc_feats    = CenterCrop([H, W])(enc_feats)

        return enc_feats

    
    def forward(self, x, enc_feats):

        for idx in range(len(self.channels) - 1):

            x        = self.up_convs[idx](x)
            enc_feat = self.crop(enc_feats[idx], x)

            ## encoder output과 decoder를 합쳐주는 부분.
            x        = torch.cat( [x, enc_feat], dim = 1)
            x        = self.dec_blocks[idx](x)

        return x


class UNet(Module):

    ## n_classes는 대체로 출력되는 segmentation mask의 channel 값으로 이용한다.
    def __init__(self,  n_classes = 1, retain_dim = True,
                 dec_channels = (64, 32, 16), enc_channels = (3, 16, 32, 64),
                 out_size     = (config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)):

        super(UNet, self).__init__()

        self.encoder = Encoder(enc_channels)
        self.decoder = Decoder(dec_channels)

        self.head       = Conv2d(dec_channels[-1], n_classes, 1)
        self.retain_dim = retain_dim
        self.out_size   = out_size


    def forward(self, x):

        enc_feats = self.encoder(x)
        dec_feats = self.decoder(enc_feats[::-1][0],
                                 enc_feats[::-1][1:])

        map = self.head(dec_feats)
        if self.retain_dim: map = F.interpolate(map, self.out_size)

        return map
                 

