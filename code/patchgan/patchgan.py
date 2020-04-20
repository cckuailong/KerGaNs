# Name: 
#   Patch Generative Adversarial Nets
# Desc:
#   DCGAN + Patch Discriminator
# Procedure:
#
#  Real images ----------------------|
#                                    |       -----      |-----|
#                                    | ----> | D | ---->| N*N |
#            -----                   |       --|--      |-----|
#  Noise --> | G | --> Fake images --|         |
#            --|--                             |
#              |<------------------------------|


from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import math


class GAN:
    def __init__(self, img_shape, sample_shape=(5,5), latent_dim=128, g_optimizer=Adam(0.0002, 0.5), d_optimizer=Adam(0.0002, 0.5), g_loss='binary_crossentropy', d_loss='mse'):
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
        
        if type(latent_dim) == int and latent_dim > 0:
            self.latent_dim = latent_dim
        else:
            print("[Error] Param 'latent' should be a positive integer, eg. 128")
            sys.exit(1)
        
        # Num of Discriminator Filters
        self.df = 64
        # Calculate output shape of D (PatchGAN)
        patch = math.ceil(self.img_shape[0] / 2)
        self.disc_patch = (patch, patch, 1)

        # Build and compile the discriminator
        self.discriminator = Discriminator(self.img_shape, self.df).modelling()
        self.discriminator.compile(loss=d_loss, optimizer=d_optimizer, metrics=['accuracy'])

        # Get generator model
        self.generator = Generator(self.img_shape, self.latent_dim).modelling()

        self.combined = self.combine()
        # Build and Compile the Combined (Generator + Discriminator)
        self.combined.compile(loss=g_loss, optimizer=g_optimizer)

    def combine(self):
        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        return Model(z, validity)



    def train_one_epoch(self, X_train, epoch, batch_size, valid, fake):
        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random batch of images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        # Generate noise randomly
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

        # Generate a batch of new images
        gen_imgs = self.generator.predict(noise)

        # Train the discriminator
        d_loss_real = self.discriminator.train_on_batch(imgs, valid)
        d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------

        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

        # Train the generator (to have the discriminator label samples as valid)
        g_loss = self.combined.train_on_batch(noise, valid)

        # Plot the progress
        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))


    # Train the Models(G && D)
    def train(self, data, epochs, batch_size=128, sample_interval=200):
        X_train = data
        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(1, epochs+1):
            self.train_one_epoch(X_train, epoch, batch_size, valid, fake)

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    # Generate images with Generator in current epoch
    def sample_images(self, epoch):
        # images matrix scale is r*c
        r, c = self.sample_shape
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        samples = "images/"
        if not os.path.exists(samples):
            os.mkdir(samples)

        # Draw and Save the images
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("%s%d.png" % (samples, epoch))
        plt.close()

class Generator:
    def __init__(self, img_shape, latent):
        self.img_shape = img_shape
        self.latent_dim = latent

    # Build Generator Model
    def modelling(self):
        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.img_shape[2], kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(inputs=noise, outputs=img)


class Discriminator:
    def __init__(self, img_shape, df):
        self.img_shape = img_shape
        self.df = df

    def d_layer(self, layer_input, filters, f_size=4, stride=2, bn=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size, strides=stride, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    # Build Discriminator Model
    def modelling(self):
        img = Input(shape=self.img_shape)

        d1 = self.d_layer(img, self.df, stride=1, bn=False)
        d2 = self.d_layer(d1, self.df*2, stride=1)
        d3 = self.d_layer(d2, self.df*4, stride=1)
        # d4 = self.d_layer(d3, self.df*8, stride=1)

        validity = Conv2D(1, kernel_size=4, strides=2, padding='same')(d3)

        model = Model(inputs=img, outputs=validity)
        model.summary()

        return model


if __name__ == "__main__":
    # Load the dataset
    (X_train, _), (_, _) = mnist.load_data()
    gan = GAN(img_shape=(28,28,1))
    gan.train(data=X_train, epochs=8000, batch_size=32, sample_interval=200)
