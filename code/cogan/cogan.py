# Name: 
#   Coupled Generative Adversarial Nets
# Desc:
#   Image1 --> Image2 Corresponding Transformer
#   E.g. Image Style migration
#   Combine two GANs, and keep the variables and weights same at the end of G, the beginning of D
# Procedure:
#
# Real images 1 ---------------------------->|
#                                            |      ------     |--> 1
#                                            | ---> | D1 | --> |
#                 ------                     |      --|---     |--> 0
#           |---->| G1 | --> Fake images 1 --|        |
#           |     --|---                              |
# Noise --> |       |<--------------------------------|
#           |     --|---                              |
#           |---->| G2 | --> Fake images 2 --|        |
#                 ------                     |      --|---     |--->1
#                                            | ---> | D2 | --> |
#                                            |      ------     |--->0
# Real images 2 ---------------------------->|


from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import scipy


class COGAN:
    def __init__(self, img_shape, sample_shape=(4,4), latent=128, g_optimizer=Adam(0.0002, 0.5), d_optimizer=Adam(0.0002, 0.5), g_loss=['binary_crossentropy', 'binary_crossentropy'], d_loss='binary_crossentropy'):
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
        discriminator = Discriminator(self.img_shape)
        self.discriminator1, self.discriminator2 = discriminator.model1, discriminator.model2
        self.discriminator1.compile(loss=d_loss, optimizer=d_optimizer, metrics=['accuracy'])
        self.discriminator2.compile(loss=d_loss, optimizer=d_optimizer, metrics=['accuracy'])


        combined = Generator(self.img_shape, self.latent_dim, self.discriminator1, self.discriminator2)
        # Build the generator
        self.generator1, self.generator2 = combined.generator1, combined.generator2

        # Build the Combined (Generator + Discriminator)
        self.combined = combined.model
        self.combined.compile(loss=g_loss, optimizer=g_optimizer)

    def train_one_epoch(self, X1, X2, epoch, batch_size, valid, fake):
        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random batch of images
        idx = np.random.randint(0, X1.shape[0], batch_size)
        imgs1 = X1[idx]
        imgs2 = X2[idx]

        # Sample noise as generator input
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

        # Generate a batch of new images
        gen_imgs1 = self.generator1.predict(noise)
        gen_imgs2 = self.generator2.predict(noise)

        # Train the discriminators
        d1_loss_real = self.discriminator1.train_on_batch(imgs1, valid)
        d2_loss_real = self.discriminator2.train_on_batch(imgs2, valid)
        d1_loss_fake = self.discriminator1.train_on_batch(gen_imgs1, fake)
        d2_loss_fake = self.discriminator2.train_on_batch(gen_imgs2, fake)
        d1_loss = 0.5 * np.add(d1_loss_real, d1_loss_fake)
        d2_loss = 0.5 * np.add(d2_loss_real, d2_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------

        g_loss = self.combined.train_on_batch(noise, [valid, valid])

        # Plot the progress
        print ("%d [D1 loss: %f, acc.: %.2f%%] [D2 loss: %f, acc.: %.2f%%] [G loss: %f]" \
            % (epoch, d1_loss[0], 100*d1_loss[1], d2_loss[0], 100*d2_loss[1], g_loss[0]))


    # Train the Models(G && D)
    def train(self, data, epochs, batch_size=128, sample_interval=200):
        X_train = data
        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        # Images in domain A and B (rotated)
        X1 = X_train[:int(X_train.shape[0]/2)]
        X2 = X_train[int(X_train.shape[0]/2):]
        X2 = scipy.ndimage.interpolation.rotate(X2, 90, axes=(1, 2))

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(1, epochs+1):
            self.train_one_epoch(X1, X2, epoch, batch_size, valid, fake)

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    # Generate images with Generator in current epoch
    def sample_images(self, epoch):
        # images matrix scale is r*c
        r, c = self.sample_shape
        noise = np.random.normal(0, 1, (r * int(c/2), self.latent_dim))
        gen_imgs1 = self.generator1.predict(noise)
        gen_imgs2 = self.generator2.predict(noise)

        gen_imgs = np.concatenate([gen_imgs1, gen_imgs2])

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
    def __init__(self, img_shape, latent, discrim_model1, discrim_model2):
        self.img_shape = img_shape
        self.latent_dim = latent

        self.generator1, self.generator2 = self.modelling()
        self.model = self.build(discrim_model1, discrim_model2)

    
    def build(self, discriminator1, discriminator2):
        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img1 = self.generator1(z)
        img2 = self.generator2(z)

        # For the combined model we will only train the generators
        discriminator1.trainable = False
        discriminator2.trainable = False

        # The valid takes generated images as input and determines validity
        valid1 = discriminator1(img1)
        valid2 = discriminator2(img2)

        # The combined model  (stacked generators and discriminators)
        # Trains generators to fool discriminators
        combined = Model(z, [valid1, valid2])

        return combined

    # Build Generator Model
    def modelling(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        noise = Input(shape=(self.latent_dim,))
        feature_repr = model(noise)

        # Generator 1
        g1 = Dense(1024)(feature_repr)
        g1 = LeakyReLU(alpha=0.2)(g1)
        g1 = BatchNormalization(momentum=0.8)(g1)
        g1 = Dense(np.prod(self.img_shape), activation='tanh')(g1)
        img1 = Reshape(self.img_shape)(g1)

        # Generator 2
        g2 = Dense(1024)(feature_repr)
        g2 = LeakyReLU(alpha=0.2)(g2)
        g2 = BatchNormalization(momentum=0.8)(g2)
        g2 = Dense(np.prod(self.img_shape), activation='tanh')(g2)
        img2 = Reshape(self.img_shape)(g2)

        model.summary()

        return Model(inputs=noise, outputs=img1), Model(inputs=noise, outputs=img2)


class Discriminator:
    def __init__(self, img_shape):
        self.img_shape = img_shape

        self.model1, self.model2 = self.build()

    def build(self):
        return self.modelling()

    # Build Discriminator Model
    def modelling(self):
        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))

        img1 = Input(shape=self.img_shape)
        img2 = Input(shape=self.img_shape)

        img1_embedding = model(img1)
        img2_embedding = model(img2)

        # Discriminator 1
        validity1 = Dense(1, activation='sigmoid')(img1_embedding)
        # Discriminator 2
        validity2 = Dense(1, activation='sigmoid')(img2_embedding)

        return Model(inputs=img1, outputs=validity1), Model(inputs=img2, outputs=validity2)

if __name__ == "__main__":
    # Load the dataset
    (X_train, _), (_, _) = mnist.load_data()
    cogan = COGAN(img_shape=(28,28,1))
    cogan.train(data=X_train, epochs=4000, batch_size=32, sample_interval=4000)