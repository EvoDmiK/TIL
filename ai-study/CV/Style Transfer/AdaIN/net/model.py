from torch.nn import functional as F
import torch.nn as nn

from misc.utils import AdaIN as adain
from misc.utils import get_mean_std


class Decoder(nn.Module):

    def __init__(self):

        super(Decoder, self).__init__()

        self.decoder = nn.Sequential()
        parameters   = [
                        [512, 256,  True, True],
                        [256, 256, False, True],
                        [256, 256, False, True],
                        [256, 256, False, True],
                        [256, 128,  True, True],
                        [128, 128, False, True],
                        [128,  64,  True, True],
                        [ 64,  64, False, True],
                        [ 64,   3, False, False]
                       ]
        
        for parameter in parameters:

            in_chn, out_chn, upsample, relu = parameter
            self.decoder += self.decoder_block(in_chn, out_chn, upsample, relu)

        
    def decoder_block(self, in_chn, out_chn, upsample = False, relu = True,
                      p_input = (1, 1, 1, 1), kernel_size = (3, 3), scale_factor = 2):



        block = nn.Sequential(
                    nn.ReflectionPad2d(p_input),
                    nn.Conv2d(in_chn, out_chn, kernel_size = kernel_size),
                )

        if     relu: block += nn.Sequential(nn.ReLU())
        if upsample: block += nn.Sequential(nn.Upsample(scale_factor = scale_factor, mode = 'nearest'))

        return block


    def forward(self, x):

        return self.decoder(x)


class Encoder(nn.Module):

    def __init__(self):

        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 3, (1, 1)))
        parameters   = [
                            [  3,  64, False],
                            [ 64,  64,  True],
                            [ 64, 128, False],
                            [128, 128,  True],
                            [128, 256, False],
                            [256, 256, False],
                            [256, 256, False],
                            [256, 256,  True],
                            [256, 512, False],
                            [512, 512, False],
                            [512, 512, False],
                            [512, 512,  True],
                            [512, 512, False],
                            [512, 512, False],
                            [512, 512, False],
                            [512, 512, False]
                        ]

        for parameter in parameters:

            in_chn, out_chn, pooling = parameter
            self.encoder += self.encoder_block(in_chn, out_chn, pooling)


    def encoder_block(self, in_chn, out_chn, pooling = False,
                      p_input = (1, 1, 1, 1), kernel_size = (3, 3)):

        block = nn.Sequential(
                    nn.ReflectionPad2d(p_input),
                    nn.Conv2d(in_chn, out_chn, kernel_size = kernel_size),
                    nn.ReLU()
                )

        if pooling: block += nn.Sequential(nn.MaxPool2d((2, 2), (2, 2), (0,0), ceil_mode = True))

        return block


    def forward(self, x):

        return self.encoder(x)


class Net(nn.Module):

    def __init__(self, encoder, decoder):

        super(Net, self).__init__()
        
        enc_layers = list(list(encoder.children())[0].children())
        self.enc_1 = nn.Sequential(*enc_layers[  :  4])
        self.enc_2 = nn.Sequential(*enc_layers[ 4: 11])
        self.enc_3 = nn.Sequential(*enc_layers[11: 18])
        self.enc_4 = nn.Sequential(*enc_layers[18: 31])

        self.decoder   = decoder
        self.loss_func = nn.MSELoss()

        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False


    
    def encode_style(self, input):

        results = [input]
        for idx in range(1, 5):
            func = getattr(self, f'enc_{idx}')
            results.append(func(results[-1]))

        return results[1:]


    def encode_content(self, input):

        for idx in range(1, 5):
            input = getattr(self, f'enc_{idx}')(input)

        return input


    def calc_content_loss(self, input, target):

        return self.loss_func(input, target)


    def calc_style_loss(self, input, target):

        input_mean ,  input_std =  get_mean_std(input)
        target_mean, target_std = get_mean_std(target)

        return self.loss_func(input_mean, input_std) + self.loss_func(target_mean, target_std)


    def forward(self, content, style, alpha = 1.0):

        style_feats   =     self.encode_style(style)
        content_feats = self.encode_content(content)

        t   = adain(content_feats, style_feats[-1])
        t   = alpha * t + (1 - alpha) * content_feats
        
        g_t       = self.decoder(t)
        g_t_feats = self.encode_style(g_t)

        loss_c    =           self.calc_content_loss(g_t_feats[-1], t)
        loss_s    = self.calc_style_loss(g_t_feats[0], style_feats[0])

        for idx in range(1, 4):

            loss_s += self.calc_style_loss(g_t_feats[idx], style_feats[idx])


        return loss_c, loss_s
    
