# Name: 
#   Super Resolution Generative Adversarial Nets
# Desc:
#   Train the (LR Image, HR Image) to make the Low Resolution Image to High Resolution
#   Vgg19 to get the images' features. Compare the features of Real HR Image and the Fake ones which G generate.
#   D use the patchGAN Discriminator to fine every batch of the image
# Procedure:
#
#  Real HR Image --------------------------------------------------------------|
#                                                                              |       -----      -------
#                                                                              | ----> | D | ---->| N*N |
#                    -----                       -------                       |       --|--      -------
#  Real LR Image --> | G | --> Fake HR Image --> | VGG | --> Fake Features --> |         |
#                    --|--                       -------                                 |
#                      |<----------------------------------------------------------------|


from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.layers import BatchNormalization, Activation, Add
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG19

import matplotlib.pyplot as plt
import sys
import os
import numpy as np

from data_loader import DataLoader


class SRGAN:
    def __init__(self, lr_img_shape, hr_img_shape, sample_shape=(2,2), g_optimizer=Adam(0.0002, 0.5), d_optimizer=Adam(0.0002, 0.5), g_loss=['binary_crossentropy', 'mse'], d_loss='mse'):
        if type(lr_img_shape) == tuple and len(lr_img_shape) == 3:
            self.lr_img_shape = lr_img_shape
        else:
            print("[Error] Param 'lr_img_shape' should be a triple set, eg. (28,28,1)")
            sys.exit(1)
        
        if type(hr_img_shape) == tuple and len(hr_img_shape) == 3:
            self.hr_img_shape = hr_img_shape
        else:
            print("[Error] Param 'lr_img_shape' should be a triple set, eg. (28,28,1)")
            sys.exit(1)
        
        if type(sample_shape) == tuple and len(sample_shape) == 2:
            self.sample_shape = sample_shape
        else:
            print("[Error] Param 'sample_shape' should be a double set, eg. (5,5)")
            sys.exit(1)

        # Configure data loader
        self.data_loader = DataLoader(img_res=(self.hr_img_shape[0], self.hr_img_shape[1]))

        # Number of residual blocks in the generator
        self.n_residual_blocks = 16

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        # Calculate output shape of D (PatchGAN)
        patch = int(self.hr_img_shape[0] / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Build and compile the vgg19
        self.vgg = VGG(self.hr_img_shape).modelling()
        self.vgg.trainable = False
        self.vgg.compile(loss='mse', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

        # Build and compile the discriminator
        self.discriminator = Discriminator(self.hr_img_shape, self.df).modelling()
        self.discriminator.compile(loss=d_loss, optimizer=d_optimizer, metrics=['accuracy'])

        # Build the generator
        self.generator = Generator(self.lr_img_shape, self.gf, self.n_residual_blocks).modelling()

        # Build the Combined (Generator + Discriminator)
        self.combined = self.combine()
        self.combined.compile(loss=g_loss, loss_weights=[1e-3, 1], optimizer=g_optimizer)

    def combine(self):
        # High res. and low res. images
        img_hr = Input(shape=self.hr_img_shape)
        img_lr = Input(shape=self.lr_img_shape)

        # Generate high res. version from low res.
        fake_hr = self.generator(img_lr)

        # Extract image features of the generated img
        fake_features = self.vgg(fake_hr)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminator determines validity of generated high res. images
        validity = self.discriminator(fake_hr)

        combined = Model([img_lr, img_hr], [validity, fake_features])

        return combined

    def train_one_epoch(self,epoch, batch_size, valid, fake):
        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Sample images and their conditioning counterparts
        imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

        # From low res. image generate high res. version
        fake_hr = self.generator.predict(imgs_lr)

        # Train the discriminators (original images = real / generated = Fake)
        d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid)
        d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------

        # Sample images and their conditioning counterparts
        imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

        # Extract ground truth image features using pre-trained VGG19 model
        image_features = self.vgg.predict(imgs_hr)

        # Train the generators
        g_loss = self.combined.train_on_batch([imgs_lr, imgs_hr], [valid, image_features])

        # Plot the progress
        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))


    # Train the Models(G && D)
    def train(self, epochs, batch_size=128, sample_interval=200):
        # Adversarial ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(1, epochs+1):
            self.train_one_epoch(epoch, batch_size, valid, fake)

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    # Generate images with Generator in current epoch
    def sample_images(self, epoch):
        samples = "images/"
        if not os.path.exists(samples):
            os.mkdir(samples)

        r,c = self.sample_shape
        
        imgs_hr, imgs_lr = self.data_loader.load_data(batch_size=2, is_testing=True)
        fake_hr = self.generator.predict(imgs_lr)

        # Rescale images 0 - 1
        imgs_lr = 0.5 * imgs_lr + 0.5
        fake_hr = 0.5 * fake_hr + 0.5
        imgs_hr = 0.5 * imgs_hr + 0.5

        # Save generated images and the high resolution originals
        titles = ['Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            for col, image in enumerate([fake_hr, imgs_hr]):
                axs[row, col].imshow(image[row])
                axs[row, col].set_title(titles[col])
                axs[row, col].axis('off')
            cnt += 1
        fig.savefig("%s%d.png" % (samples, epoch))
        plt.close()

        # Save low resolution images for comparison
        for i in range(r):
            fig = plt.figure()
            plt.imshow(imgs_lr[i])
            fig.savefig('%s%d_lowres%d.png' % (samples, epoch, i))
            plt.close()

class VGG:
    def __init__(self, hr_img_shape):
        self.hr_img_shape = hr_img_shape

    def modelling(self):
        """
        Builds a pre-trained VGG19 model that outputs image features extracted at the
        third block of the model
        """
        vgg = VGG19(weights="imagenet")
        vgg.outputs = [vgg.layers[9].output]

        img = Input(shape=self.hr_img_shape)

        # Extract image features
        img_features = vgg(img)

        return Model(img, img_features)

class Generator:
    def __init__(self, lr_img_shape, gf, n_residual_blocks):
        self.lr_img_shape = lr_img_shape
        self.gf = gf
        self.n_residual_blocks = n_residual_blocks

    def residual_block(self, layer_input, filters):
        """Residual block described in paper"""
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
        d = Activation('relu')(d)
        d = BatchNormalization(momentum=0.8)(d)
        d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
        d = BatchNormalization(momentum=0.8)(d)
        d = Add()([d, layer_input])
        return d

    def deconv2d(self, layer_input):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
        u = Activation('relu')(u)
        return u

    def modelling(self):
        # Low resolution image input
        img_lr = Input(shape=self.lr_img_shape)

        # Pre-residual block
        c1 = Conv2D(64, kernel_size=9, strides=1, padding='same')(img_lr)
        c1 = Activation('relu')(c1)

        # Propogate through residual blocks
        r = self.residual_block(c1, self.gf)
        for _ in range(self.n_residual_blocks - 1):
            r = self.residual_block(r, self.gf)

        # Post-residual block
        c2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
        c2 = BatchNormalization(momentum=0.8)(c2)
        c2 = Add()([c2, c1])

        # Upsampling
        u1 = self.deconv2d(c2)
        u2 = self.deconv2d(u1)

        # Generate high resolution output
        gen_hr = Conv2D(self.lr_img_shape[2], kernel_size=9, strides=1, padding='same', activation='tanh')(u2)

        return Model(inputs=img_lr, outputs=gen_hr)


class Discriminator:
    def __init__(self, hr_img_shape, df):
        self.hr_img_shape = hr_img_shape
        self.df = df

    def d_block(self, layer_input, filters, strides=1, bn=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    def modelling(self):
        # Input img
        d0 = Input(shape=self.hr_img_shape)

        d1 = self.d_block(d0, self.df, bn=False)
        d2 = self.d_block(d1, self.df, strides=2)
        d3 = self.d_block(d2, self.df*2)
        d4 = self.d_block(d3, self.df*2, strides=2)
        d5 = self.d_block(d4, self.df*4)
        d6 = self.d_block(d5, self.df*4, strides=2)
        d7 = self.d_block(d6, self.df*8)
        d8 = self.d_block(d7, self.df*8, strides=2)

        d9 = Dense(self.df*16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='sigmoid')(d10)

        return Model(inputs=d0, outputs=validity)


if __name__ == "__main__":
    # Load the dataset
    srgan = SRGAN(lr_img_shape=(64, 64, 3), hr_img_shape=(256, 256, 3))
    srgan.train(epochs=900, batch_size=1, sample_interval=300)