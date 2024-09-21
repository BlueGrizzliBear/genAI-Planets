# Set the size of the random noise vector
latent_dim = 100

# Build the discriminator
img_dim = 256
img_shape = (img_dim, img_dim, 3)

# Set the learning rate for the optimizer
learning_rate = 0.00002
# Set the value for the Adam optimizer beta_1 parameter
beta_1 = 0.5

# Set the number of epochs and batch size for training
epochs = 5000
batch_size = 8

# Generator rows and columns
rows = 4
cols = 4