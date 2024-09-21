from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam

from constants import learning_rate, beta_1

def build_gan(generator, discriminator, latent_dim):
    print('Start building GAN')

    # Make the discriminator non-trainable during the generator training
    discriminator.trainable = False
    
    # Connect the generator and discriminator
    gan_input = Input(shape=(latent_dim,))
    generated_img = generator(gan_input)
    gan_output = discriminator(generated_img)
    
    # Create the GAN model
    gan = Model(gan_input, gan_output)
    
    gan.summary()
    print('Finished building GAN')

    print('Compiling the GAN')
    gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate, beta_1=beta_1), metrics=['accuracy'])
    print('Finished compiling the GAN')

    return gan
