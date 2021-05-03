#DCGAN for generating UC-Birds dataset

## Basic setting
* Strided Convolution & Striden Transpose Convolution
* ReLU & LeakyReLU
* BatchNorm
* No Fully Connected Layers
* No Pooling Layers
* Weight Initialization N(0, 0,02)
* Batch =128
* ReLU & LeakyReLU for discriminator
* Adam(lr = 0.0002 momentum=0.5)

## Additional setting

* Apply multi-way loss [from Auxiliary-GAN]

##Network Architecture

### Generator
* nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)
    - tip) output_size = (input size -1) * stride - 2 * padding + (kernel_size -1) +1
* use ReLU and BatchNorm after ConvTranspose2d operation except for last ConvTranspose2d operation
* image size => [1x1 -> 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64]
* channel size => [(noise_dim + label_dim) -> 64*8 -> 64*4 -> 64*4 -> 64*2 -> 64 ->3]
* tanh after last ConvTranspose2d operation

### Discriminator
* use nn.LeakyReLU(negative_slope=0.2, inplace=True) instead of nn.ReLU
* Real_Fake Classifier & Categorical Classifier
* img_size = [64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4] -> [4x4->1x1]                  #for Real_Fake Classifier
                                                        [flatten -> num_classes]    #for Categorical Classifier
* channels = [3 -> 64 -> 64*2 =>64*4 =>64*8] => [64*8 -> 1]                         # for Real_Fake Classifier
                                             => [flatten -> num_classes]            # for Categorical Classifier