# Name: 
#   Pix2Pix Generative Adversarial Nets
# Desc:
#   Train image pair (A,B), to realize the image style migration
# Procedure:
#
#  Real imageB ---------------------------->|
#                                           |       -----      |----> 1 (real)
#       |---------------------------------->| ----> | D | ---->|
#       |          -----                    |       --|--      |----> 0 (fake)
#  Real ImageA --> | G | --> Fake imageA -->|         |
#                  --|--                              |
#                    |<-------------------------------|


from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from data_loader import DataLoader

import matplotlib.pyplot as plt
import sys
import os
import numpy as np


class PIX2PIX:
    def __init__(self, img_shape, sample_shape=(3,3), g_optimizer=Adam(0.0002, 0.5), d_optimizer=Adam(0.0002, 0.5), g_loss=['mse', 'mae'], d_loss='mse'):
        if type(img_shape) == tuple and len(img_shape) == 3:
            self.img_shape = img_shape
        else:
            print("[Error] Param 'img_shape' should be a triple set, eg. (28,28,1)")
            sys.exit(1)
        
        if type(sample_shape) == tuple and len(sample_shape) == 2:
            self.sample_shape = sample_shape
        else:
            print("[Error] Param 'sample_shape' should be a double set, eg. (5,5)")
            sys.exit(1)

        # Configure data loader
        self.dataset_name = 'facades'
        self.data_loader = DataLoader(dataset_name=self.dataset_name, img_res=(self.img_shape[0], self.img_shape[1]))

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_shape[0] / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        # Build and compile the discriminator
        self.discriminator = Discriminator(self.img_shape, self.df).modelling()
        self.discriminator.compile(loss=d_loss, optimizer=d_optimizer, metrics=['accuracy'])

        # Get generator model
        self.generator = Generator(self.img_shape, self.gf).modelling()

        self.combined = self.combine()
        # Build and Compile the Combined (Generator + Discriminator)
        self.combined.compile(loss=g_loss, loss_weights=[1, 100], optimizer=g_optimizer)

    def combine(self):
        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])

        return combined



    def train_one_epoch(self, epoch, batch_size, valid, fake):
        for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Condition on B and generate a translated version
            fake_A = self.generator.predict(imgs_B)

            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
            d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # -----------------
            #  Train Generator
            # -----------------

            # Train the generators
            g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

            # Plot the progress
            print ("[Epoch %d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f]" % (epoch,
                                                                    batch_i, self.data_loader.n_batches,
                                                                    d_loss[0], 100*d_loss[1],
                                                                    g_loss[0]))


    # Train the Models(G && D)
    def train(self, epochs, batch_size=128, sample_interval=200):
        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(1, epochs+1):
            self.train_one_epoch(epoch, batch_size, valid, fake)

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    # Generate images with Generator in current epoch
    def sample_images(self, epoch):
        r, c = self.sample_shape
        # images matrix scale is r*c
        imgs_A, imgs_B = self.data_loader.load_data(batch_size=3, is_testing=True)
        fake_A = self.generator.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        samples = "images/"
        if not os.path.exists(samples):
            os.mkdir(samples)

        # Draw and Save the images
        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("%s%d.png" % (samples, epoch))
        plt.close()

class Generator:
    """U-Net Generator"""
    def __init__(self, img_shape, gf):
        self.img_shape = img_shape
        self.gf = gf

    def conv2d(self, layer_input, filters, f_size=4, bn=True):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    def deconv2d(self, layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = BatchNormalization(momentum=0.8)(u)
        u = Concatenate()([u, skip_input])
        return u

    # Build Generator Model
    def modelling(self):
        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = self.conv2d(d0, self.gf, bn=False)
        d2 = self.conv2d(d1, self.gf*2)
        d3 = self.conv2d(d2, self.gf*4)
        d4 = self.conv2d(d3, self.gf*8)
        d5 = self.conv2d(d4, self.gf*8)
        d6 = self.conv2d(d5, self.gf*8)
        d7 = self.conv2d(d6, self.gf*8)

        # Upsampling
        u1 = self.deconv2d(d7, d6, self.gf*8)
        u2 = self.deconv2d(u1, d5, self.gf*8)
        u3 = self.deconv2d(u2, d4, self.gf*8)
        u4 = self.deconv2d(u3, d3, self.gf*4)
        u5 = self.deconv2d(u4, d2, self.gf*2)
        u6 = self.deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.img_shape[2], kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model(d0, output_img)


class Discriminator:
    def __init__(self, img_shape, df):
        self.img_shape = img_shape
        self.df = df

    def d_layer(self, layer_input, filters, f_size=4, bn=True):
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d
    
    # Build Discriminator Model
    def modelling(self):
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = self.d_layer(combined_imgs, self.df, bn=False)
        d2 = self.d_layer(d1, self.df*2)
        d3 = self.d_layer(d2, self.df*4)
        d4 = self.d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(inputs=[img_A, img_B], outputs=validity)


if __name__ == "__main__":
    # Load the dataset
    pix2pix = PIX2PIX(img_shape=(256,256,3))
    pix2pix.train(epochs=200, batch_size=1, sample_interval=200)