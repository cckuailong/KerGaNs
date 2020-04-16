# Name: 
#   Information Maximizing Generative Adversarial Nets
# Desc:
#   Interpretable Representation Learning
#   We can change the specific value in vector "C", to control the specific features
#   E.g. Digit's category, continuous, and so on
#   Recognier is like the auto encoder(VAE)
# Procedure:
#
#         |---  Real images ----------------------------|
#         |                                             |       -----      |----> 1 (real)
#    |--->|                                             | ----> | D | ---->|
#    |    |                     -----                   |       -----   |  |----> 0 (fake)
#    |    |--- C   ++ Noise --> | G | --> Fake images --|               |
#    |         |                -----          |                        |
#    |         |                -----          |                        |
#    |         <--------------- | R | <--------|                        |
#    |                          -----                                   |
#    |<-----------------------------------------------------------------|


from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import sys
import os
import numpy as np


def mutual_info_loss(c, c_given_x):
    """The mutual information metric we aim to minimize"""
    eps = 1e-8
    conditional_entropy = K.mean(- K.sum(K.log(c_given_x + eps) * c, axis=1))
    entropy = K.mean(- K.sum(K.log(c + eps) * c, axis=1))

    return conditional_entropy + entropy

class INFOGAN:
    def __init__(self, img_shape, num_classes, sample_shape=(5,5), latent=128, 
    g_optimizer=Adam(0.0002, 0.5), d_optimizer=Adam(0.0002, 0.5), r_optimizer=Adam(0.0002, 0.5),
    g_loss=['binary_crossentropy', mutual_info_loss], d_loss=['binary_crossentropy'], r_loss=[mutual_info_loss]):
        if type(img_shape) == tuple and len(img_shape) == 3:
            self.img_shape = img_shape
        else:
            print("[Error] Param 'img_shape' should be a triple set, eg. (28,28,1)")
            sys.exit(1)
        
        if type(num_classes) == int and num_classes > 0:
            self.num_classes = num_classes
        else:
            print("[Error] Param 'num_classes' should be a positive integer, eg. 128")
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
        discriminator = Discriminator(self.img_shape, self.num_classes)
        self.discriminator = discriminator.model
        self.discriminator.compile(loss=d_loss, optimizer=d_optimizer, metrics=['accuracy'])

        # Build and Compile the recognizer
        self.recognizer = discriminator.recognizer
        self.recognizer.compile(loss=r_loss, optimizer=r_optimizer, metrics=['accuracy'])

        combined = Generator(self.img_shape, self.latent_dim, self.discriminator, self.recognizer)
        # Build the generator
        self.generator = combined.generator

        # Build the Combined (Generator + Discriminator)
        self.combined = combined.model
        self.combined.compile(loss=g_loss, optimizer=g_optimizer)

    def train_one_epoch(self, X_train, epoch, batch_size, valid, fake):
        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random batch of images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        # Generate noise and C randomly
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim-self.num_classes))
        c = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)
        # One hot
        c = to_categorical(c, num_classes=self.num_classes)
        # concat noise and C
        gen_input = np.concatenate((noise, c), axis=1)


        # Generate a batch of new images
        gen_imgs = self.generator.predict(gen_input)

        # Train the discriminator
        d_loss_real = self.discriminator.train_on_batch(imgs, valid)
        d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------

        noise = np.random.normal(0, 1, (batch_size, self.latent_dim-self.num_classes))

        # Train the generator (to have the discriminator label samples as valid)
        g_loss = self.combined.train_on_batch(gen_input, [valid, c])

        # Plot the progress
        print ("%d [D loss: %.2f, acc.: %.2f%%] [Q loss: %.2f] [G loss: %.2f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[1], g_loss[2]))


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

class Generator:
    def __init__(self, img_shape, latent, discrim_model, recog_model):
        self.img_shape = img_shape
        self.latent_dim = latent

        self.generator = self.modelling()
        self.model = self.build(discrim_model, recog_model)

    
    def build(self, discrim_model, recog_model):
        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # Get the Discriminator Model
        discriminator = discrim_model
        # For the combined model we will only train the generator
        discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = discriminator(img)
        # The recognition network produces the label
        target_label = recog_model(img)
        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        combined = Model(z, [validity, target_label])

        return combined

    # Build Generator Model
    def modelling(self):
        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(self.img_shape[2], kernel_size=3, padding='same'))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(inputs=noise, outputs=img)


class Discriminator:
    def __init__(self, img_shape, num_classes):
        self.img_shape = img_shape
        self.num_classes = num_classes

        self.recognizer = None
        self.model = self.build()

    def build(self):
        discriminator, self.recognizer = self.modelling()
        return discriminator

    # Build Discriminator Model
    def modelling(self):
        model = Sequential()

        model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(512, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Flatten())

        model.summary()

        img = Input(shape=self.img_shape)
        img_embedding = model(img)

        # Discriminator
        validity = Dense(1, activation='sigmoid')(img_embedding)

        # Recognition
        q_net = Dense(128, activation='relu')(img_embedding)
        label = Dense(self.num_classes, activation='softmax')(q_net)

        # Return discriminator and recognition network
        return Model(inputs=img, outputs=validity), Model(inputs=img, outputs=label)


if __name__ == "__main__":
    # Load the dataset
    (X_train, _), (_, _) = mnist.load_data()
    infogan = INFOGAN(img_shape=(28,28,1), num_classes=10)
    infogan.train(data=X_train, epochs=900, batch_size=32, sample_interval=300)