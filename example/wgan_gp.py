# Name: 
#   Wasserstein Generative Adversarial Nets With Gradient Penalty
# Desc:
#   Improvement of WGAN, use gradient penalty instead of weight clipping
#   Get the medium distribution between P_real and P_fake
# Procedure:
#
#  Real images ----------------------|
#                                    |       -----      |----> 1 (real)
#                                    | ----> | D | ---->|
#            -----                   |       --|--      |----> 0 (fake)
#  Noise --> | G | --> Fake images --|         |
#            --|--                             |
#              |<------------------------------|

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

import tensorflow.keras.backend as K
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import pickle


class WGAN_GP:
    def __init__(self, img_shape, sample_shape=(5,5), latent=128):
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
        
        # Following parameter and optimizer set as recommended in paper
        self.n_discriminator = 5

        # Init the generator and discriminator model
        if os.path.isfile("result/sample/g.h5"):
            self.gen_model = load_model("result/sample/g.h5")
        else:
            self.gen_model = Generator(self.img_shape, self.latent_dim).modelling()
        if os.path.isfile("result/sample/d.h5"):
            self.discrim_model = load_model("result/sample/d.h5")
        else:
            self.discrim_model = Discriminator(self.img_shape, self.latent_dim).modelling()

        # Build the discriminator
        discriminator = Discriminator(self.img_shape, self.latent_dim, self.gen_model, self.discrim_model)
        self.discriminator_train_func = discriminator.build()

        # Build the generator
        combined = Generator(self.img_shape, self.latent_dim, self.gen_model, self.discrim_model)
        self.generator = combined.generator

        # Build the Combined (Generator + Discriminator)
        self.combined_train_func = combined.build()


    def train_one_epoch(self, X_train, epoch, batch_size, valid, fake, dummy):
        # ---------------------
        #  Train Discriminator
        # ---------------------

        for _ in range(self.n_discriminator):
            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Generate noise randomly
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            # Get the alpha
            alpha = np.random.uniform(size=(batch_size, 1, 1 ,1))
            # Train the discriminator
            d_loss_real, d_loss_fake = self.discriminator_train_func([imgs, noise, alpha])
            d_loss = d_loss_real - d_loss_fake

        # ---------------------
        #  Train Generator
        # ---------------------

        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        # Train the generator (to have the discriminator label samples as valid)
        g_loss, = self.combined_train_func([noise])

        # Plot the progress
        print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss, g_loss))


    # Train the Models(G && D)
    def train(self, data, epochs, batch_size=128, sample_interval=200):
        X_train = data
        # Rescale -1 to 1
        #X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        #X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid =-np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty

        for epoch in range(1, epochs+1):
            self.train_one_epoch(X_train, epoch, batch_size, valid, fake, dummy)

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                self.gen_model.save("result/model/g.h5")
                self.discrim_model.save("result/model/d.h5")

    # Generate images with Generator in current epoch
    def sample_images(self, epoch):
        # images matrix scale is r*c
        r, c = self.sample_shape
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5*gen_imgs+0.5

        samples = "result/sample/"
        if not os.path.exists(samples):
            os.mkdir(samples)

        # Draw and Save the images
        fig, axs = plt.subplots(r, c, figsize=(6.4, 6.4))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                #print(gen_imgs[cnt, :,:,:].shape)
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("%s%d.png" % (samples, epoch))
        plt.close()


class Generator:
    def __init__(self, img_shape=None, latent=None, gen_model=None, discrim_model=None):
        self.img_shape = img_shape
        self.latent_dim = latent

        self.generator = gen_model
        self.discriminator = discrim_model

    
    def build(self):
        # Only Train generator
        self.discriminator.trainable = False
        self.generator.trainable = True

        noise = Input(shape=(self.latent_dim,))
        # Get Fake image with noise
        z = self.generator(noise)
        # generator loss function
        loss = -K.mean(self.discriminator(z))
        
        # Build the train strategy
        training_updates = Adam(lr=0.0001, beta_1=0.0, beta_2=0.9).get_updates(params=self.generator.trainable_weights, loss=loss)
        train_func = K.function([noise], [loss], training_updates)

        return train_func

    # Build Generator Model
    def modelling(self):
        model = Sequential()

        model.add(Dense(128 * 64 * 64, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((64, 64, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.img_shape[2], kernel_size=4, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(inputs=noise, outputs=img)


class Discriminator:
    def __init__(self, img_shape=None, latent=None, gen_model=None, discrim_model=None):
        self.img_shape = img_shape
        self.latent_dim = latent

        self.generator = gen_model
        self.discriminator = discrim_model

    # Get the Interpolated, In the middle of Real and Fake
    def get_interpolated(self, real_img, fake_img):
        alpha = K.placeholder(shape=(None,1,1,1))
        interpolated_img = Input(shape=self.img_shape, 
                                tensor=alpha*real_img + (1-alpha)*fake_img)

        return interpolated_img, alpha

    # Gradient Penalty Calculation
    def gradient_penalty_loss(self, real_img, fake_img, interpolated_img):
        loss_real = K.mean(self.discriminator(real_img))
        loss_fake = K.mean(self.discriminator(fake_img))

        grad_mixed = K.gradients(self.discriminator(interpolated_img), [interpolated_img])[0]
        gradients_sqr = K.square(grad_mixed)
        norm_grad_mixed = K.sqrt(K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape))))
        grad_penalty = K.mean(K.square(norm_grad_mixed-1))

        loss = loss_fake - loss_real + 10 * grad_penalty

        return loss_real, loss_fake, loss

    def build(self):
        # Only train discriminator
        self.generator.trainable = False
        self.discriminator.trainable = True

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)

        # Noise input
        noise = Input(shape=(self.latent_dim,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator(noise)

        # Construct weighted average between real and fake images
        interpolated_img,alpha = self.get_interpolated(real_img, fake_img)

        # Get loss of Gradient Penalty
        loss_real, loss_fake, loss = self.gradient_penalty_loss(real_img, fake_img, interpolated_img)

        # Build the train strategy
        training_updates = Adam(lr=0.0001, beta_1=0.0, beta_2=0.9).get_updates(params=self.discriminator.trainable_weights, loss=loss)
        discriminator_train = K.function([real_img, noise, alpha],
                                [loss_real, loss_fake],    
                                training_updates)

        return discriminator_train

    # Build Discriminator Model
    def modelling(self):
        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(inputs=img, outputs=validity)


if __name__ == "__main__":
    # Load the dataset
    with open("/dev/shm/data.pkl", "rb") as f:
        data = pickle.load(f)
    
    wgan_gp = WGAN_GP(img_shape=(256,256,3), sample_shape=(2,2))
    wgan_gp.train(data=data, epochs=6000, batch_size=8, sample_interval=100)
