# Name: 
#   Bidirectional Generative Adversarial Nets
# Desc:
#   GAN + VAE's encoder, train with (z, img) pair
# Procedure:
#     |--------------------------------------->|
#     |             -----                      |---->|
#  Real images ---> | Q | ---> Real latent --->|     |
#     |             -----                            |
#     |                                              |       -----      |----> 1 (real)
#     |-------------->|                              | ----> | D | ---->|
#                   --|--                            |       -----      |----> 0 (fake)
#  Noise ---------> | G | --> Fake images ---->|     |
#     |             -----                      |---->|
#     |--------------------------------------->|
#

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import sys
import os
import numpy as np


class BIGAN:
    def __init__(self, img_shape, sample_shape=(5,5), latent_dim=128, g_optimizer=Adam(0.0002, 0.5), d_optimizer=Adam(0.0002, 0.5), g_loss=['binary_crossentropy', 'binary_crossentropy'], d_loss=['binary_crossentropy']):
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

        # Build and compile the discriminator
        self.discriminator = Discriminator(self.img_shape, self.latent_dim).modelling()
        self.discriminator.compile(loss=d_loss, optimizer=d_optimizer, metrics=['accuracy'])

        # Build the encoder
        self.encoder = Encoder(self.img_shape, self.latent_dim).modelling()

        # Build the generator
        self.generator = Generator(self.img_shape, self.latent_dim).modelling()

        # Build the Combined (Generator + Discriminator)
        self.combined = self.combine()
        self.combined.compile(loss=g_loss, optimizer=g_optimizer)

    def combine(self):
        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Generate image from sampled noise
        fake_z = Input(shape=(self.latent_dim, ))
        fake_img = self.generator(fake_z)

        # Encode real image
        real_img = Input(shape=self.img_shape)
        real_z = self.encoder(real_img)

        # Discrminate the (real_z, real_img), (fake_z, fake_img)
        fake = self.discriminator([fake_z, fake_img])
        valid = self.discriminator([real_z, real_img])

        # Set up and compile the combined model
        # Trains generator to fool the discriminator
        combined = Model([fake_z, real_img], [fake, valid])

        return combined

    def train_one_epoch(self, X_train, epoch, batch_size, valid, fake):
        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Sample noise and generate img
        fake_z = np.random.normal(size=(batch_size, self.latent_dim))
        fake_imgs = self.generator.predict(fake_z)

        # Select a random batch of images and encode
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_imgs = X_train[idx]
        real_z = self.encoder.predict(real_imgs)

        # Train the discriminator
        d_loss_real = self.discriminator.train_on_batch([real_z, real_imgs], valid)
        d_loss_fake = self.discriminator.train_on_batch([fake_z, fake_imgs], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------

        # Train the generator (to have the discriminator label samples as valid)
        g_loss = self.combined.train_on_batch([fake_z, real_imgs], [valid, fake])

        # Plot the progress
        print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))


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

class Encoder:
    def __init__(self, img_shape, latent_dim):
        self.img_shape = img_shape
        self.latent_dim = latent_dim

    def modelling(self):
        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.latent_dim))

        model.summary()

        img = Input(shape=self.img_shape)
        z = model(img)

        return Model(img, z)

class Generator:
    def __init__(self, img_shape, latent):
        self.img_shape = img_shape
        self.latent_dim = latent

    # Build Generator Model
    def modelling(self):
        model = Sequential()

        model.add(Dense(512, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(inputs=noise, outputs=img)


class Discriminator:
    def __init__(self, img_shape, latent_dim):
        self.img_shape = img_shape
        self.latent_dim = latent_dim

    # Build Discriminator Model
    def modelling(self):
        z = Input(shape=(self.latent_dim, ))
        img = Input(shape=self.img_shape)
        d_in = concatenate([z, Flatten()(img)])

        model = Dense(1024)(d_in)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        model = Dense(1024)(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = Dropout(0.5)(model)
        validity = Dense(1, activation="sigmoid")(model)

        return Model([z, img], validity)


if __name__ == "__main__":
    # Load the dataset
    (X_train, _), (_, _) = mnist.load_data()
    bigan = BIGAN(img_shape=(28,28,1))
    bigan.train(data=X_train, epochs=900, batch_size=32, sample_interval=300)
