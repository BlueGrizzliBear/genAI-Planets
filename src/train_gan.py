import numpy as np
import gc
import tensorflow as tf

from gan_builder.format_data import format_data, load_and_preprocess_image
from gan_builder.gan import build_gan
from gan_builder.discriminator import build_discriminator
from gan_builder.generator import build_generator
from gan_builder.user_validation import user_validation

from constants import latent_dim, img_dim, img_shape, learning_rate, epochs, batch_size

# Enable eager execution
tf.config.run_functions_eagerly(True)

# Enable debug mode for tf.data functions
tf.data.experimental.enable_debug_mode()

def train_step(discriminator, gan, real_images, generated_images, real_labels, fake_labels, noise):
	with tf.device('/GPU:0'):  # Explicitly set to use GPU
		# Train the discriminator on real and fake images
		discriminator.trainable=True
		d_loss_real = discriminator.train_on_batch(real_images, real_labels)
		d_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)
		d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

		# Train the generator to fool the discriminator
		for _ in range(2):  # Train generator twice for every discriminator training
			noise = tf.random.normal((batch_size, latent_dim))
			discriminator.trainable=False
			g_loss = gan.train_on_batch(noise, real_labels)
		
		return d_loss, g_loss


def main():
	
	# Check available devices
	physical_devices = tf.config.list_physical_devices('GPU')
	print("Num GPUs Available: ", len(physical_devices))
	for device in physical_devices:
		print(device)

	# Format the data -----------------------------------------------------------
	# Load the training and testing datasets
	print('Start formatting data')
	dataset_folder = './dataset/'
	train_df = format_data(dataset_folder, 'planets', img_dim)
	print('Finished formatting data')

  # Build generator | discriminator | gan -------------------------------------
	generator = build_generator(latent_dim, img_dim)
	discriminator = build_discriminator(img_shape, img_dim)
	gan = build_gan(generator, discriminator, latent_dim)

	user_response = user_validation("Do you want to proceed? [y/n]: ")
	if user_response == 'y':
		print("User chose to proceed.")
	else:
		print("User chose not to proceed.")
		return

  # Train the GAN -------------------------------------------------------------
	print('Start to train the GAN')
	# Load and preprocess all images
	train_images = np.array([load_and_preprocess_image(path) for path in train_df['image_path'].values])

	# Set the label for real and fake images
	real_labels = np.ones(batch_size)
	fake_labels = np.zeros(batch_size)

	try:
		for epoch in range(epochs):
			# Select a random batch of images
			idx = tf.random.uniform((batch_size,), minval=0, maxval=train_images.shape[0], dtype=tf.int32)
			real_images = tf.gather(train_images, idx)

			# Generate a batch of fake images
			noise = tf.random.normal((batch_size, latent_dim))
			generated_images = generator.predict(noise)

			real_images = tf.convert_to_tensor(real_images)
			real_labels = tf.convert_to_tensor(real_labels)
			generated_images = tf.convert_to_tensor(generated_images)
			fake_labels = tf.convert_to_tensor(fake_labels)


			d_loss, g_loss = train_step(discriminator, gan, real_images, generated_images, real_labels, fake_labels, noise)



			# # Train the discriminator on real and fake images
			# discriminator.trainable=True
			# d_loss_real = discriminator.train_on_batch(real_images, real_labels)
			# d_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)
			# d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

			# # Train the generator to fool the discriminator
			# for _ in range(2):  # Train generator twice for every discriminator training
			# 	noise = np.random.normal(0, 1, (batch_size, latent_dim))
			# 	discriminator.trainable=False
			# 	g_loss = gan.train_on_batch(noise, real_labels)

			# Display the progress every 100 epochs
			# if epoch % 10 == 0:
			# 	print(f"Epoch {epoch}/{epochs} [D loss: {d_loss[0]:.2f} | D accuracy: {100 * d_loss[1]:.2f}] [G loss: {g_loss[0]:.2f}]")
			print(f"Epoch {epoch}/{epochs} [D loss: {d_loss[0]:.2f} | D accuracy: {100 * d_loss[1]:.2f}] [G loss: {g_loss[0]:.2f}]")

			del real_images, generated_images
			gc.collect()
		print('Finished training the GAN')
	except KeyboardInterrupt:
		print('Training interrupted. Saving the current state of the generator...')

	# Save the trained generator model
	path = f'./model/trained_generator_{img_dim}.h5'
	generator.save(path)
	print('Saved the trained GAN model at ', path)

if __name__ == "__main__":
    main()