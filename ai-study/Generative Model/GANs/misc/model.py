import torch.nn.functional as F
import torch.nn as nn
import torch


class ConvBlock(nn.Module):
    
    def __init__(self, in_, out_, kernel, stride = 1, drop_p = None, act = 'relu'):
        
        super(ConvBlock, self).__init__()
        self.drop_p = drop_p
        self.conv   = nn.Conv2d(in_, out_, kernel, stride = stride, padding = 1)
        
        if act   == 'relu':      self.act = F.relu
        elif act == 'leakyrelu': self.act = F.leaky_relu
        else:                    self.act = F.tanh
        
        
    def forward(self, x):
        
        x = self.act(self.conv(x))
        if self.drop_p != None: x = F.dropout(x, self.drop_p)
        
        return x, F.max_pool2d(x, kernel_size = 2)
        
        
class Generator(nn.Module):
    
    def __init__(self):
        super(Generator, self).__init__()
        
        self.down_conv1 = ConvBlock(3 , 32, 3, drop_p = 0.1)
        self.down_conv2 = ConvBlock(32, 64, 3)
        self.down_conv3 = ConvBlock(64, 96, 3)
        self.down_conv4 = ConvBlock(96, 96, 3)
        
        self.b_conv5    = ConvBlock(96, 96 , 3, drop_p = 0.3)
        self.b_conv6    = ConvBlock(96, 256, 3)
    
        self.concat_7   = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up_conv7   = ConvBlock(224, 128, 3)
        
        self.concat_8   = nn.ConvTranspose2d(128, 96, 2, 2)
        self.up_conv8   = ConvBlock(192, 128, 3)
        
        self.concat_9   = nn.ConvTranspose2d(128, 64, 2, 2)
        self.up_conv9   = ConvBlock(128, 128, 3)
        
        self.concat_10  = nn.ConvTranspose2d(128, 32, 2, 2)
        self.up_conv10  = ConvBlock(64, 128, 3)
        
        self.up_conv11  = ConvBlock(128, 3, 1, act = 'tanh')
        
    
    def forward(self, x):
        
        d1, f1 = self.down_conv1(x)
        d2, f2 = self.down_conv2(f1)
        d3, f3 = self.down_conv3(f2)
        d4, f4 = self.down_conv4(f3)
        
        b5,  _ = self.b_conv5(f4)
        b5,  _ = self.b_conv6(b5)
        
        u6     = self.concat_7(b5)
        u6     = torch.cat([u6, d4], dim = 1)
        u6,  _ = self.up_conv7(u6)
        
        u7     = self.concat_8(u6)
        u7     = torch.cat([u7, d3], dim = 1)
        u7, _  = self.up_conv8(u7)
        
        u8     = self.concat_9(u7)
        u8     = torch.cat([u8, d2], dim = 1)
        u8, _  = self.up_conv9(u8)
        
        u9     = self.concat_10(u8)
        u9     = torch.cat([u9, d1], dim = 1)
        u9, _  = self.up_conv10(u9)
        
        u10, _ = self.up_conv11(u9)
        return u10

    
class Discriminator(nn.Module):
    
    def __init__(self):
        
        super(Discriminator, self).__init__()
        
        self.conv1  = ConvBlock(3  , 64 , 4, 2, act = 'leakyrelu')
        self.conv2  = ConvBlock(64 , 128, 4, 2, act = 'leakyrelu')
        self.conv3  = ConvBlock(128, 256, 4, 2, act = 'leakyrelu')
        
        self.conv4  = nn.Conv2d(256, 512, 4, 1, padding = 1)
        self.conv5  = nn.Conv2d(512,   1, 3, 1)
        self.bn     = nn.BatchNorm2d(512)

        
    def forward(self, x):
        
        x, _ = self.conv1(x)
        x, _ = self.conv2(x)
        x, _ = self.conv3(x)
        x    = F.leaky_relu(self.bn(self.conv4(x)))

        return self.conv5(x)
        
        
        
        
        
        
    

        