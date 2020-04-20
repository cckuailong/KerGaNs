# Name: 
#   Dual Generative Adversarial Nets
# Desc:
#   Like CycleGAN, combine func,G, D model has some differences
#   Use Wasserstein Distance as loss(WGAN)
# Procedure:
#                                                        |----------------------------->|
#  Real ImageA -------------------------------|          |                              |
#                                             |       ---|----     |----> 1 (real)      |
#                                             | ----> | d_AB | --->|                    |
#                  --------                   |       ---|----     |----> 0 (fake)      |
#  Real ImageB --> | g_AB | --> Fake ImageB --|          |                              |
#                  ---|----                              |                              |
#                     |<---------------------------------|                              |
#                     |                                                                 |
#                     |<---------------------------------|                              |
#  Real ImageB -------------------------------|          |                              |
#                                             |       ---|----     |----> 1 (real)      |
#                                             | ----> | d_BA | --->|                    |
#                  --------                   |       ---|----     |----> 0 (fake)      |
#  Real ImageA --> | g_BA | --> Fake ImageA --|          |                              |
#                  ---|----                              |                              |
#                     |<---------------------------------|                              |
#                     |                                                                 |
#                     |<----------------------------------------------------------------|


from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from data_loader import DataLoader

import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import scipy


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

class DUALGAN:
    def __init__(self, img_shape, sample_shape=(4,4), g_optimizer=Adam(0.0002, 0.5), d_optimizer=Adam(0.0002, 0.5), g_loss=[wasserstein_loss, wasserstein_loss, 'mae', 'mae'], d_loss=wasserstein_loss):
        if type(img_shape) == tuple and len(img_shape) == 3:
            self.img_shape = img_shape
        else:
            print("[Error] Param 'img_shape' should be a triple set, eg. (28,28,1)")
            sys.exit(1)
        self.img_dim = self.img_shape[0]*img_shape[1]
        
        if type(sample_shape) == tuple and len(sample_shape) == 2:
            self.sample_shape = sample_shape
        else:
            print("[Error] Param 'sample_shape' should be a double set, eg. (5,5)")
            sys.exit(1)

        # Build and compile the discriminator
        self.d_A = Discriminator(self.img_shape).modelling()
        self.d_B = Discriminator(self.img_shape).modelling()
        self.d_A.compile(loss=d_loss, optimizer=d_optimizer, metrics=['accuracy'])
        self.d_B.compile(loss=d_loss, optimizer=d_optimizer, metrics=['accuracy'])

        # Get generator model
        self.g_AB = Generator(self.img_shape).modelling()
        self.g_BA = Generator(self.img_shape).modelling()

        self.combined = self.combine()
        # Build and Compile the Combined (Generator + Discriminator)
        self.combined.compile(loss=g_loss,
                            loss_weights=[1, 1, 100, 100],
                            optimizer=g_optimizer)
    def combine(self):
        # Input images from both domains
        img_A = Input(shape=self.img_dim)
        img_B = Input(shape=self.img_dim)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        combined = Model(inputs=[img_A, img_B],
                        outputs=[ valid_A, valid_B,
                                reconstr_A, reconstr_B])

        return combined

    def sample_generator_input(self, X, batch_size):
        # Sample random batch of images from X
        idx = np.random.randint(0, X.shape[0], batch_size)
        return X[idx]

    def train_one_epoch(self, X_A, X_B, epoch, batch_size, valid, fake, clip_value, n_critic):
        for i in range(n_critic):
            # ----------------------
            #  Train Discriminators
            # ----------------------

            # Sample generator inputs
            imgs_A = self.sample_generator_input(X_A, batch_size)
            imgs_B = self.sample_generator_input(X_B, batch_size)

            # Translate images to their opposite domain
            fake_B = self.g_AB.predict(imgs_A)
            fake_A = self.g_BA.predict(imgs_B)

            # Train the discriminators
            d_A_loss_real = self.d_A.train_on_batch(imgs_A, valid)
            d_A_loss_fake = self.d_A.train_on_batch(fake_A, fake)

            d_B_loss_real = self.d_B.train_on_batch(imgs_B, valid)
            d_B_loss_fake = self.d_B.train_on_batch(fake_B, fake)

            d_A_loss = 0.5 * np.add(d_A_loss_real, d_A_loss_fake)
            d_B_loss = 0.5 * np.add(d_B_loss_real, d_B_loss_fake)

            # Clip discriminator weights
            for d in [self.d_A, self.d_B]:
                for l in d.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                    l.set_weights(weights)

            # ------------------
            #  Train Generators
            # ------------------

            # Train the generators
            g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, valid, imgs_A, imgs_B])

            # Plot the progress
            print ("[Epoch: %d] [N: %d] [D1 loss: %f] [D2 loss: %f] [G loss: %f]" \
                % (epoch, i, d_A_loss[0], d_B_loss[0], g_loss[0]))


    # Train the Models(G && D)
    def train(self, data, epochs, batch_size=128, sample_interval=200):
        # Rescale -1 to 1
        X_train = (data.astype(np.float32) - 127.5) / 127.5

        # Domain A and B (rotated)
        X_A = X_train[:int(X_train.shape[0]/2)]
        X_B = scipy.ndimage.interpolation.rotate(X_train[int(X_train.shape[0]/2):], 90, axes=(1, 2))

        X_A = X_A.reshape(X_A.shape[0], self.img_dim)
        X_B = X_B.reshape(X_B.shape[0], self.img_dim)

        clip_value = 0.01
        n_critic = 4

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(1, epochs+1):
            self.train_one_epoch(X_A, X_B, epoch, batch_size, valid, fake, clip_value, n_critic)

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch, X_A, X_B)

    # Generate images with Generator in current epoch
    def sample_images(self, epoch, X_A, X_B):
        r, c = self.sample_shape

        # Sample generator inputs
        imgs_A = self.sample_generator_input(X_A, c)
        imgs_B = self.sample_generator_input(X_B, c)

        # Images translated to their opposite domain
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)

        gen_imgs = np.concatenate([imgs_A, fake_B, imgs_B, fake_A])
        gen_imgs = gen_imgs.reshape((r, c, self.img_shape[0], self.img_shape[1], self.img_shape[2]))

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
                axs[i,j].imshow(gen_imgs[i, j, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("%s%d.png" % (samples, epoch))
        plt.close()

class Generator:
    def __init__(self, img_shape):
        self.img_shape = img_shape
        self.img_dim = self.img_shape[0]*self.img_shape[1]

    # Build Generator Model
    def modelling(self):
        X = Input(shape=(self.img_dim,))

        model = Sequential()
        model.add(Dense(256, input_dim=self.img_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.4))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.4))
        model.add(Dense(self.img_dim, activation='tanh'))

        X_translated = model(X)

        return Model(inputs=X, outputs=X_translated)


class Discriminator:
    def __init__(self, img_shape):
        self.img_shape = img_shape
        self.img_dim = self.img_shape[0]*self.img_shape[1]
    
    # Build Discriminator Model
    def modelling(self):
        img = Input(shape=(self.img_dim,))

        model = Sequential()
        model.add(Dense(512, input_dim=self.img_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1))

        validity = model(img)

        return Model(inputs=img, outputs=validity)


if __name__ == "__main__":
    # Load the dataset
    (X_train, _), (_, _) = mnist.load_data()
    dualgan = DUALGAN(img_shape=(28,28,1))
    dualgan.train(data=X_train, epochs=200, batch_size=32, sample_interval=200)