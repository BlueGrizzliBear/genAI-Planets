from keras.models import Sequential
from keras.layers import Input, Dense, Reshape, Conv2D, LeakyReLU, BatchNormalization, UpSampling2D

# Generator model
def build_generator(latent_dim, img_dim):
    print('Start building generator')

    node = int(img_dim / 8)

    # Initializing the neural network
    generator = Sequential()
    
    # Adding an input layer to the network
    generator.add(Dense(units=256 * node * node, input_dim=latent_dim))
    generator.add(LeakyReLU(0.2))
    generator.add(Reshape((node, node, 256)))
    
    # Upsampling layers
    generator.add(UpSampling2D())
    generator.add(Conv2D(128, kernel_size=3, padding='same'))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization(momentum=0.8))
    
    generator.add(UpSampling2D())
    generator.add(Conv2D(64, kernel_size=3, padding='same'))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization(momentum=0.8))
    
    # Additional upsampling and convolutional layers
    generator.add(UpSampling2D())
    generator.add(Conv2D(32, kernel_size=3, padding='same'))
    generator.add(LeakyReLU(0.2))
    generator.add(BatchNormalization(momentum=0.8))

    # Output layer
    generator.add(Conv2D(3, kernel_size=3, padding='same', activation='tanh'))

    generator.summary()

    print('Finished building generator')

    return generator
