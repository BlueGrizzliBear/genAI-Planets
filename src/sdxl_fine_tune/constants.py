# Description: Constants for the fine-tuning process.

# Set the file name to the local model file
model_file_name='sdXL_v10VAEFix.safetensors'

# Set the rank for the LoRA layer
rank = 4

# Set the path to the image directory
image_dir='dataset/planets'

# Set the target directory for the images used to fine tune
target_dir = '/dedicated_dataset/planets/'

# Default prompt template (you can customize this)
# default_prompt = "A beautiful planet in space with {planet_name} features."
default_prompt = "A planet seen from space"

# Set the learning rate for the optimizer
learning_rate = 5e-5

# Set the number of epochs and batch size for training
epochs = 10
save_steps=100
batch_size = 2
gradient_accumulation_steps = 2
