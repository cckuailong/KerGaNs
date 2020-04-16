# KerGaNs

Various GAN via Keras

## List

### GAN

- Name:
  Generative Adversarial Nets
- Desc:
  Basic GAN

### DCGAN

- Name:
    Deep Convolutions Generative Adversarial Network
- Desc:
    Generator and Discriminator use convolutions
        1. Using convolution layer instead of pooling layer
        2. Remove the full connection layer
        3. Use Batch normalization
        4. Use appropriate activation function

### WGAN

- Name:
    Wasserstein Generative Adversarial Nets
- Desc:
    Use Wasserstein Distance as the loss function to solve the unstablity of GAN
    Wasserstein Distance: Also named EarthMover Distance, The energy cost of moving mound P1 to P2
    The Discriminator in WGAN don't use sigmoid, because the normalization function has a defect that 
    the convergence is slow at both ends of the function, and the gradient is almost 0
        1. Entirely solve the unstablity of GAN
        2. Mainly solve the model collapse of GAN
        3. Have a centain target to instruct the training
        4. No need to cost plenty of time to design the Nerual Net

### WGAN-GP

- Name:
    Wasserstein Generative Adversarial Nets With Gradient Penalty
- Desc:
    Improvement of WGAN, use gradient penalty instead of weight clipping
    Get the medium distribution between P_real and P_fake

### CGAN

- Name:
    Conditional Generative Adversarial Nets
- Desc:
    GAN with labels, Input a label, generate corresponding image
    Exsample：Input: 1      Get: Image of digit 1

### SGAN

- Name:
    Semi-Supervised Generative Adversarial Nets
- Desc:
    1. Semi-Supervised Learning with GAN, classifier combined with discriminator
    2. G and D/C are adversarial, D and C are mutually reinforcing
    3. Features D learned can be useful to the C

### ACGAN

- Name:
    Auxiliary Classifier Generative Adversarial Nets
- Desc:
    SGAN + CGAN

### InfoGAN

- Name:
    Information Maximizing Generative Adversarial Nets
- Desc:
    Interpretable Representation Learning
    We can change the specific value in vector "C", to control the specific features
    E.g. Digit's category, continuous, and so on
    Recognier is like the auto encoder(VAE)

### AAE

- Name:
    Adversarial Autoencoder
- Desc:
    VAE + GAN
