from diffusers import StableDiffusionXLPipeline
import loralib as lora
import torch.nn as nn
from transformers import Trainer, TrainingArguments
import torch
import pandas as pd
import numpy as np

from load_data import load_dataset_with_metadata

from constants import model_file_name, rank, image_dir, learning_rate, epochs, batch_size, save_steps, gradient_accumulation_steps, default_prompt

def main():
    # Load the model from a local directory
    local_model_path = f"./model/{model_file_name}"  # Path to your locally saved SDXL model
    pipeline = StableDiffusionXLPipeline.from_pretrained(local_model_path, torch_dtype=torch.float16)
    pipeline.to("cuda")  # Move the model to GPU

    # Apply LoRA to attention layers in UNet
    def apply_lora_to_unet(unet):
        for name, module in unet.named_modules():
            if isinstance(module, nn.Linear):  # You can also target nn.Conv2d or other layers as needed
                lora.Linear(module, r=rank)  # Apply LoRA with rank 4, adjust r based on the needs
        return unet

    # Apply LoRA to the UNet
    pipeline.unet = apply_lora_to_unet(pipeline.unet)

    # Load your local dataset with labeled images
    dataset = load_dataset_with_metadata(image_dir)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./model/sdxl-lora-finetuned",
        learning_rate=learning_rate,
        num_train_epochs=epochs,  # You may increase this for better performance
        per_device_train_batch_size=batch_size,  # Adjust based on your GPU memory
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=True,  # Use mixed precision
        save_steps=save_steps,
        logging_dir="./logs",
    )

    # Custom Trainer class to handle the LoRA fine-tuning
    class LoRAFineTuner(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            outputs = model(**inputs)
            loss = outputs.loss
            return (loss, outputs) if return_outputs else loss

    # Initialize the Trainer
    trainer = LoRAFineTuner(
        model=pipeline.unet,  # Fine-tune the modified UNet with LoRA applied
        args=training_args,
        train_dataset=dataset,  # The dataset of images and text prompts
        tokenizer=pipeline.tokenizer,  # Tokenizer to process the prompts
    )

    # Start training
    trainer.train()


    # Generate an image using the fine-tuned model
    prompt = default_prompt
    generated_image = pipeline(prompt).images[0]

    # Save the generated image
    generated_image.save("./lora_finetuned_image.png")

    # Save the fine-tuned model to disk
    pipeline.save_pretrained("./model/sdxl-lora-finetuned")


if __name__ == "__main__":
    main()