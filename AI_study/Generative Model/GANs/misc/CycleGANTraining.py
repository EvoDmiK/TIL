from tensorflow.keras import Model
import tensorflow as tf


class CycleGANTraining(Model):
    
    def __init__(self, gen_G, disc_X, gen_F, disc_Y, **kwargs):
        
        super().__init__(**kwargs)
        self.gen_G   = gen_G
        self.gen_F   = gen_F
        self.disc_X  = disc_X
        self.disc_Y  = disc_Y
        
        
    def compile(self, g_optimG, d_optimX, g_optimF, d_optimY, bceLoss):
        
        super().compile()
        self.g_optimG = g_optimG
        self.g_optimF = g_optimF
        
        self.d_optimX = d_optimX
        self.d_optimY = d_optimY
        
        self.bceLoss  = bceLoss
        
        
    def train_step(self, images):
        
        (input_image, target_image) = images
        with tf.GradientTape() as genG_tape, tf.GradientTape() as discX_tape, \
             tf.GradientTape() as genF_tape, tf.GradientTape() as discY_tape:
            
            
            gen_imagesY   = self.gen_G(input_image, training = True)
            cycled_imageX = self.gen_F(gen_imagesY, training = True)
            
            gen_imagesX   = self.gen_F(target_image, training = True)
            cycled_imageY = self.gen_G(gen_imagesX  , training = True)
            
            samegenX      = self.gen_F(input_image,  training = True)
            samegenY      = self.gen_G(target_image, training = True)
            
            
            disc_real_outY = self.disc_Y([target_image], training = True)
            disc_fake_outY = self.disc_Y([gen_imagesY] , training = True)
            
            disc_real_outX = self.disc_X([input_image] , training = True)
            disc_fake_outX = self.disc_X([gen_imagesX] , training = True)
            
            
            lossA = 10 * (tf.reduce_mean(tf.abs(target_image - cycled_imageY)))
            lossB = 10 * (tf.reduce_mean(tf.abs(input_image  - cycled_imageX)))
            
            total_cycle_loss = lossA + lossB
            
            identityLossG    = 10 * 0.5 * (tf.reduce_mean(tf.abs(target_image - samegenY)))
            identityLossF    = 10 * 0.5 * (tf.reduce_mean(tf.abs(input_image - samegenX)))
            
            ganLossG         = self.bceLoss(tf.ones_like(disc_fake_outY), disc_fake_outY)
            ganLossF         = self.bceLoss(tf.ones_like(disc_fake_outX), disc_fake_outX)
            
            real_disc_lossY  = self.bceLoss(tf.ones_like(disc_real_outY), disc_real_outY)
            fake_disc_lossY  = self.bceLoss(tf.zeros_like(disc_fake_outY), disc_fake_outY)
            
            real_disc_lossX  = self.bceLoss(tf.ones_like(disc_real_outX), disc_real_outX)
            fake_disc_lossX  = self.bceLoss(tf.zeros_like(disc_fake_outX), disc_fake_outX)
            
            
            total_disc_lossY = 0.5 * (real_disc_lossY + fake_disc_lossY)
            total_disc_lossX = 0.5 * (real_disc_lossX + fake_disc_lossX)
            
            total_gen_lossG  = ganLossG + total_cycle_loss + identityLossG
            total_gen_lossF  = ganLossF + total_cycle_loss + identityLossF
            
            
        genGrad_G  = genG_tape.gradient(total_gen_lossG, self.gen_G.trainable_variables)
        genGrad_F  = genF_tape.gradient(total_gen_lossF, self.gen_F.trainable_variables)
        
        discGrad_X = discX_tape.gradient(total_disc_lossX, self.disc_X.trainable_variables)
        discGrad_Y = discY_tape.gradient(total_disc_lossY, self.disc_Y.trainable_variables)
        
        
        self.g_optimG.apply_gradients(zip(genGrad_G, self.gen_G.trainable_variables))
        self.g_optimF.apply_gradients(zip(genGrad_F, self.gen_F.trainable_variables))
        
        self.d_optimX.apply_gradients(zip(discGrad_X, self.disc_X.trainable_variables))
        self.d_optimY.apply_gradients(zip(discGrad_Y, self.disc_Y.trainable_variables))
        
        return {'disc_lossX' : total_disc_lossX, 'gen_lossG' : ganLossG + total_cycle_loss,
                'disc_lossY' : total_disc_lossY, 'gen_lossF' : ganLossF + total_cycle_loss}