# Project Title

This repository contains scripts and instructions for fine-tuning the SDXL model with LoRA, as well as managing the conda environment.

## Overview

The main functionalities of this repository include:

1. **Fine-Tuning the SDXL Model with LoRA**: Scripts to fine-tune the Stable Diffusion XL (SDXL) model using Low-Rank Adaptation (LoRA).
2. **Conda Environment Management**: Instructions for exporting and importing conda environments using a `requirements.txt` file or an `environment.yml` file.

## Fine-Tuning the SDXL Model with LoRA

### Prerequisites

- Python 3.x
- Required Python packages: `torch`, `diffusers`, `loralib`, `transformers`, `pandas`, `Pillow`

### Usage

1. **Prepare the Dataset**: Ensure your dataset is organized and accessible.
2. **Run the Training Script**: Execute the training script to fine-tune the SDXL model.

Example command to run the training script:

```sh
python train.py
```

### Training Script

The training script (train.py) performs the following steps:

1. Loads the SDXL model and applies LoRA.
2. Loads the dataset and prepares it for training.
3. Configures the training arguments and initializes the Trainer.
4. Starts the training process.
5. Generates an image using the fine-tuned model and saves it.
6. Saves the fine-tuned model to disk.


### Conda Environment Management

This repository contains instructions for exporting and importing conda environments using a `requirements.txt` file.

#### Exporting the Conda Environment

To export the current conda environment to a `requirements.txt` file, follow these steps:

1. Open your terminal.
2. Activate your conda environment:

    ```sh
    conda activate your_env_name
    ```

3. Export the environment to a `requirements.txt` file:

    ```sh
    conda list --export > requirements.txt
    ```

#### Installing Dependencies from `requirements.txt`

To install dependencies from the `requirements.txt` file, follow these steps:

1. Create a new conda environment (optional):

    ```sh
    conda create --name new_env_name python=3.x
    ```

2. Activate the new environment:

    ```sh
    conda activate new_env_name
    ```

3. Install the dependencies from the `requirements.txt` file:

    ```sh
    pip install -r requirements.txt
    ```

#### Alternative: Using `environment.yml`

Alternatively, you can use `conda env export` to create an `environment.yml` file, which is more suitable for conda environments:

1. Export the environment to an `environment.yml` file:

    ```sh
    conda env export > environment.yml
    ```

2. Create a new environment from this file:

    ```sh
    conda env create -f environment.yml
    ```

This method ensures that all conda-specific packages and dependencies are correctly handled.

## License

This project is licensed under the MIT License.