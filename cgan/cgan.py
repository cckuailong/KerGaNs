# Name: 
#   Conditional Generative Adversarial Nets
# Desc:
#   GAN with labels
# Procedure:
#
#         |---  Real images --------------------------------------|
#         |                                                       |                          -----      |----> 1 (real)
#    |--->|                                                       |-- +++ --Real Labels----> | D | ---->|
#    |    |                               -----                   |                          -----   |  |----> 0 (fake)
#    |    |---  Noise +++ Rand Labels --> | G | --> Fake images --|                                  |
#    |                                    -----                                                      |
#    |<--------------  ------------------------------------------------------------------------------|


from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from tensorflow.keras.layers import BatchNormalization, Embedding
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import sys
import os
import numpy as np


class GAN:
    def __init__(self, img_shape, num_classes, sample_shape=(2,5), latent=128, g_optimizer=Adam(0.0002, 0.5), d_optimizer=Adam(0.0002, 0.5), g_loss='binary_crossentropy', d_loss='binary_crossentropy'):
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

        if np.prod(sample_shape) != num_classes:
            print("[Error] Param 'sample_shape''s scale should be same as Param 'num_classes', eg. (2,5) -- 10")
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

        combined = Generator(self.img_shape, self.num_classes, self.latent_dim, self.discriminator)
        # Build the generator
        self.generator = combined.generator

        # Build the Combined (Generator + Discriminator)
        self.combined = combined.model
        self.combined.compile(loss=g_loss, optimizer=g_optimizer)

    def train_one_epoch(self, X_train, y_train, epoch, batch_size, valid, fake):
        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Select a random batch of images
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs, labels = X_train[idx], y_train[idx]

        # Generate noise randomly
        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

        # Generate a batch of new images
        gen_imgs = self.generator.predict([noise, labels])

        # Train the discriminator
        d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
        d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------

        noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

        # Condition on labels
        rand_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)

        # Train the generator (to have the discriminator label samples as valid)
        g_loss = self.combined.train_on_batch([noise, rand_labels], valid)

        # Plot the progress
        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))


    # Train the Models(G && D)
    def train(self, data, epochs, batch_size=128, sample_interval=200):
        X_train, y_train = data[0], data[1]
        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        y_train = y_train.reshape(-1, 1)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(1, epochs+1):
            self.train_one_epoch(X_train, y_train, epoch, batch_size, valid, fake)

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    # Generate images with Generator in current epoch
    def sample_images(self, epoch):
        # images matrix scale is r*c
        r, c = self.sample_shape
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        rand_labels = np.arange(0, self.num_classes).reshape(-1, 1)
        gen_imgs = self.generator.predict([noise, rand_labels])

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
    def __init__(self, img_shape, num_classes, latent, discrim_model):
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.latent_dim = latent

        self.generator = self.modelling()
        self.model = self.build(discrim_model)

    
    def build(self, discrim_model):
        # The generator takes noise as input and generates imgs
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # Get the Discriminator Model
        discriminator = discrim_model
        # For the combined model we will only train the generator
        discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = discriminator([img, label])

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        combined = Model([noise, label], validity)

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
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model(inputs=[noise, label], outputs=img)


class Discriminator:
    def __init__(self, img_shape, num_classes):
        self.img_shape = img_shape
        self.num_classes = num_classes

        self.model = self.build()

    def build(self):
        return self.modelling()

    # Build Discriminator Model
    def modelling(self):
        model = Sequential()

        model.add(Dense(512, input_dim=np.prod(self.img_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        flat_img = Flatten()(img)

        model_input = multiply([flat_img, label_embedding])

        validity = model(model_input)

        return Model(inputs=[img, label], outputs=validity)


if __name__ == "__main__":
    # Load the dataset
    data, (_, _) = mnist.load_data()
    gan = GAN(img_shape=(28,28,1), num_classes=10)
    gan.train(data=data, epochs=3000, batch_size=32, sample_interval=1000)