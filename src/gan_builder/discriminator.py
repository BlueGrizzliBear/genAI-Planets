from keras.models import Sequential
from keras.layers import Dropout, Input, Dense, Flatten, Conv2D, LeakyReLU, BatchNormalization
from keras.optimizers import Adam

from constants import learning_rate, beta_1

# Discriminator model
def build_discriminator(img_shape, img_dim):
    print('Start building discriminator')

    #initializing a neural network
    discriminator = Sequential()
    discriminator.add(Input(shape=img_shape))

    # Adding convolutional layers
    discriminator.add(Conv2D(img_dim, kernel_size=3, strides=2, padding="same"))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.25))

    discriminator.add(Conv2D(img_dim*2, kernel_size=3, strides=2, padding="same"))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.25))


    # Adding dense layers
    # discriminator.add(Dense(units=img_dim))
    # discriminator.add(LeakyReLU(0.2))
    # discriminator.add(Dropout(0.2))

    # Flatten the output
    discriminator.add(Flatten())

    # discriminator.add(Dense(units=int(img_dim/2)))
    # discriminator.add(LeakyReLU(0.2))
    # discriminator.add(Dropout(0.3))

    # discriminator.add(Dense(units=int(img_dim/4)))
    # discriminator.add(LeakyReLU(0.2))
    # discriminator.add(Dropout(0.3))

    # Output layer
    discriminator.add(Dense(1, activation='sigmoid'))
    
    discriminator.summary()
    print('Finished building discriminator')

    print('Start compiling discriminator')
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=learning_rate, beta_1=beta_1), metrics=['accuracy'])
    print('Finished compiling discriminator')

    return discriminator
