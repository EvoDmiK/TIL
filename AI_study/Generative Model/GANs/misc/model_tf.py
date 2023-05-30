from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras import Model
from tensorflow.keras import Input


class CycleGAN:
    
    def __init__(self, width, height):
        
        self.width  = width
        self.height = height
        
        
    def conv_block(self, input_, channel, kernel, stride = 1, drop_p = None, activation = 'relu'):
        
        if activation.lower() != 'leakyrelu':
            conv  = Conv2D(channel, (kernel, kernel), strides = stride, activation = activation, 
                            padding = 'same')(input_)
            
        else:
            conv  = Conv2D(channel, (kernel, kernel), strides = stride, padding = 'same')(input_)
            conv  = LeakyReLU()(conv)
        
        if drop_p != None: conv = Dropout(drop_p)(conv)
        
        return conv, MaxPool2D((2, 2))(conv)
    
    
    
    def conv_concatenate(self, input_, concat_, channel, kernel, stride, drop_p = None):
        
        conv_t = Conv2DTranspose(channel, (kernel, kernel), strides = (stride, stride),
                                 padding = 'same')(input_)
        concat = concatenate([conv_t, concat_])
        if drop_p != None: concat = Dropout(drop_p)(concat)
        
        return concat
                        
        
    def generator(self):
        
        inputs = Input([self.height, self.width, 3])
        
        ## Down sampling 레이어
        d1,  f1 = self.conv_block(inputs, 32, 3, drop_p = 0.1)
        d2,  f2 = self.conv_block(f1    , 64, 3)
        d3,  f3 = self.conv_block(f2    , 96, 3)
        d4,  f4 = self.conv_block(f3    , 96, 3)
        
        ## U-Net의 가운데 부분 (양의 이차 함수의 최소값 부분)
        b5,   _ = self.conv_block(f4,  96, 3, drop_p = 0.3)
        b5,   _ = self.conv_block(b5, 256, 3)
        
        ## Up sampling 레이어
        u6      = self.conv_concatenate(b5, d4, 128, 2, 2)
        u6, _   = self.conv_block(u6, 128, 3)
        
        u7      = self.conv_concatenate(u6, d3,  96, 2, 2)
        u7, _   = self.conv_block(u7, 128, 3)
        
        u8      = self.conv_concatenate(u7, d2, 64, 2, 2)
        u8, _   = self.conv_block(u8, 128, 3)
        
        u9      = self.conv_concatenate(u8, d1, 32, 2, 2, drop_p = 0.1)
        u9, _   = self.conv_block(u9, 128, 3)
        
        ## 출력 레이어
        outputs = self.conv_block(u9, 3, 1, activation = 'tanh')
        
        return Model(inputs, outputs)
        
    
    def discriminator(self):
        
        target_image = Input(
                                shape = [self.height, self.width, 3],
                                name  = 'target_image'
                            )
        
        
        x, _    = self.conv_block(target_image,  64, 4, strides = 2, activation = 'LeakyReLU')
        x, _    = self.conv_block(           x, 128, 4, strides = 2, activation = 'LeakyReLU')
        x, _    = self.conv_block(           x, 256, 4, strides = 2, activation = 'LeakyReLU')
        
        x       = Conv2D(512, 4, strides = 1, padding = 'same')(x)
        x       = BatchNormalization()(x)
        x       = LeakyReLU()(x)
        outputs = Conv2D(1, 3, strides = 1)(x)
        
        return Model(inputs = [targetImage], outputs = outputs)