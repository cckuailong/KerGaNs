# KerGaNs

Various GAN via Keras

## Generate Image

### GAN

- Name:
    Generative Adversarial Nets
- Desc:
    Basic GAN

### BGAN

- Name:
    Boundary Seeking GAN
- Desc:
    Loss function pay attention to the loss of D, which is usually 0.5

### LSGAN

- Name:
    Least Square Generative Adversarial Nets
- Desc:
    Basic GAN + least square loss

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

### BIGAN

- Name:
    Bidirectional Generative Adversarial Nets
- Desc:
    GAN + VAE's encoder, train with (z, img) pair

## Image Enhancement

### SRGAN

- Name:
    Super Resolution Generative Adversarial Nets
- Desc:
    Train the (LR Image, HR Image) to make the Low Resolution Image to High Resolution
    Vgg19 to get the images' features. Compare the features of Real HR Image and the Fake ones which G generate.
    D use the patchGAN Discriminator to fine every batch of the image

## Image Style Migration

### Pix2Pix

- Name:
    Pix2Pix Generative Adversarial Nets
- Desc:
    Train image pair (A,B), to realize the image style migration

### CycleGAN

- Name:
    Cycle Generative Adversarial Nets
- Desc:
    Like two combined Pix2Pix(U-Net Gennerator + Patch Discriminator)

### DiscoGan

- Name:
    Disco Generative Adversarial Nets
- Desc:
    Like CycleGAN, combine func has some differences

### DualGAN

- Name:
    Dual Generative Adversarial Nets
- Desc:
    Like CycleGAN, combine func,G, D model has some differences
    Use Wasserstein Distance as loss(WGAN)
