# Name: 
#   Adversarial Autoencoder
# Desc:
#   VAE + GAN
# Procedure:
#                    |<---------|<--------------------|
#                    |          |                     |
#                  --|--      --|--                   |
# Real images ---> | Q | ---> | P |                   |
#                  -----  |   -----                 --|--      |----> 1 (real)
#                         |-- real latent --| ----> | D | ---->|
#                                           |       -----      |----> 0 (fake)
# Noise Latent ---------------------------->|
#


from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Lambda
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import sys
import os
import numpy as np


class AAE:
    def __init__(self, img_shape, sample_shape=(5,5), latent=10, g_optimizer=Adam(0.0002, 0.5), d_optimizer=Adam(0.0002, 0.5), g_loss=['mse', 'binary_crossentropy'], d_loss='binary_crossentropy'):
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
        
        if type(latent) == int and latent > 0:
            self.latent_dim = latent
        else:
            print("[Error] Param 'latent' should be a positive integer, eg. 128")
            sys.exit(1)

        # Build and compile the discriminator
        discriminator = Discriminator(self.img_shape, self.latent_dim)
        self.discriminator = discriminator.model
        self.discriminator.compile(loss=d_loss, optimizer=d_optimizer, metrics=['accuracy'])

        vae = Generator(self.img_shape, self.latent_dim, self.discriminator)
        # Build the encoder and decoder
        self.encoder = vae.encoder
        self.decoder = vae.decoder

        # Build the Combined (Generator + Discriminator)
        self.vae = vae.model
        self.vae.compile(loss=g_loss, loss_weights=[0.999, 0.001], optimizer=g_optimizer)

    def train_one_epoch(self, X_train, epoch, batch_size, valid, fake):
        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random batch of images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        latent_real = self.encoder.predict(imgs)
        latent_fake = np.random.normal(size=(batch_size, self.latent_dim))

        # Train the discriminator
        d_loss_real = self.discriminator.train_on_batch(latent_real, valid)
        d_loss_fake = self.discriminator.train_on_batch(latent_fake, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------

        # Train the generator
        g_loss = self.vae.train_on_batch(imgs, [imgs, valid])

        # Plot the progress
        print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))

    # Train the Models(G && D)
    def train(self, data, epochs, batch_size=128, sample_interval=200):
        X_train = data
        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

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
        gen_imgs = self.decoder.predict(noise)

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
    def __init__(self, img_shape, latent, discrim_model):
        self.img_shape = img_shape
        self.latent_dim = latent

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.discriminator = discrim_model
        self.model = self.build(discrim_model)

    
    def build(self, discrim_model):
        img = Input(shape=self.img_shape)
        # The generator takes the image, encodes it and reconstructs it
        # from the encoding
        encoded_repr = self.encoder(img)
        reconstructed_img = self.decoder(encoded_repr)

        # For the adversarial_autoencoder model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator determines validity of the encoding
        validity = self.discriminator(encoded_repr)

        # The adversarial_autoencoder model  (stacked generator and discriminator)
        adversarial_autoencoder = Model(img, [reconstructed_img, validity])

        return adversarial_autoencoder

    def sampling(self, args): 
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)), mean=0.)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def build_encoder(self):
        img = Input(shape=self.img_shape)
        h = Flatten()(img)
        h = Dense(512)(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = Dense(512)(h)
        h = LeakyReLU(alpha=0.2)(h)
        z_mean = Dense(self.latent_dim)(h)
        z_log_var = Dense(self.latent_dim)(h)

        latent_repr = Lambda(self.sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])

        return Model(img, latent_repr)

    def build_decoder(self):
        model = Sequential()

        model.add(Dense(512, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        z = Input(shape=(self.latent_dim,))
        img = model(z)

        return Model(z, img)

class Discriminator:
    def __init__(self, img_shape, latent_dim):
        self.img_shape = img_shape
        self.latent_dim = latent_dim

        self.model = self.build()

    def build(self):
        return self.modelling()

    # Build Discriminator Model
    def modelling(self):
        model = Sequential()

        model.add(Dense(512, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation="sigmoid"))
        model.summary()

        encoded_repr = Input(shape=(self.latent_dim, ))
        validity = model(encoded_repr)

        return Model(inputs=encoded_repr, outputs=validity)


if __name__ == "__main__":
    # Load the dataset
    (X_train, _), (_, _) = mnist.load_data()
    aae = AAE(img_shape=(28,28,1))
    aae.train(data=X_train, epochs=4000, batch_size=32, sample_interval=4000)