from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.callbacks import Callback
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
import tensorflow as tf

def get_train_monitor(test_input, test_output, image_path, batch_size, epoch_interval):
    
    input_image  = next(iter(test_input))
    output_image = next(iter(test_output))
    
    class TrainMonitor(Callback):
        
        def __init__(self, epoch_interval = None):
            
            self.epoch_interval = epoch_interval
            
            
        def on_epoch_end(self, epoch, logs = None):
            
            if self.epoch_interval and epoch % self.epoch_interval == 0:
                
                preds = self.model.gen_G.predict(inut_image)
                (fig, axes) = subplots(nrows   = batch_size, ncols = 3,
                                       figsize = (50, 50))
                
                
                for (ax, inp, pred, target) in zip(axes, input_image, preds, output_image):
                    
                    ax[0].imshow(array_to_img(inp))
                    ax[0].set_title('input image')
                    
                    ax[1].imshow(array_to_img(pred))
                    ax[1].set_title('predicted image')
                    
                    ax[2].imshow(array_to_img(target))
                    ax[2].set_title('target image')
                    
                plt.savefig(f'{image_path}/{epoch:03d}.png')
                plt.close()
                
    train_monitor = TrainMonitor(epoch_interval = epoch_interval)
    return train_monitor