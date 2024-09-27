# Generate an image using the fine-tuned model
prompt = "A surreal landscape with glowing mountains"
generated_image = pipeline(prompt).images[0]

# Save the generated image
generated_image.save("lora_finetuned_image.png")
