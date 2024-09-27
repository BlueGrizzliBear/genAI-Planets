import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

from constants import latent_dim, rows, cols

# Function to generate and display images
def generate_images(generator, rows, cols, latent_dim):
    noise = np.random.normal(0, 1, (rows * cols, latent_dim))
    generated_images = generator.predict(noise)

    # Rescale images from [-1, 1] to [0, 1]
    generated_images = 0.5 * generated_images + 0.5

    # Display generated images
    fig, axs = plt.subplots(rows, cols, figsize=(5, 5))
    cnt = 0
    for i in range(rows):
        for j in range(cols):
            # axs[i, j].imshow(generated_images[cnt, :, :, 0], cmap='gray')
            axs[i, j].imshow(generated_images[cnt])
            axs[i, j].axis('off')
            cnt += 1
    plt.show()

def main():
    generator = load_model('./model/trained_generator_256.h5', compile=False)
    # Generate and display images using the trained generator
    generate_images(generator, rows, cols, latent_dim)

if __name__ == "__main__":
    main()