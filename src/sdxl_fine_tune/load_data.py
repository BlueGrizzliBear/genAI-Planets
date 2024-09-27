import os
import piexif
import shutil
from PIL import Image
from datasets import Dataset

from constants import default_prompt, target_dir

# Function to add a prompt to an image's EXIF metadata
def add_prompt_to_metadata(image_path, prompt):
    img = Image.open(image_path)

    # Load existing EXIF data (or create an empty one if none exists)
    exif_dict = piexif.load(img.info.get('exif', b''))
    
    # Add prompt to the "ImageDescription" tag in the EXIF data
    user_comment = prompt.encode("utf-8")
    exif_dict["0th"][piexif.ImageIFD.ImageDescription] = user_comment
    
    # Convert the EXIF data back into bytes
    exif_bytes = piexif.dump(exif_dict)
    
    # Save the image with updated EXIF data
    img.save(image_path, "jpeg", exif=exif_bytes)


# Function to read the prompt (description) from the image's EXIF metadata
def read_prompt_from_metadata(image_path):
    img = Image.open(image_path)
    
    # Load EXIF data
    exif_dict = piexif.load(img.info.get('exif', b''))
    
    # Read the "ImageDescription" field (prompt)
    prompt = exif_dict["0th"].get(piexif.ImageIFD.ImageDescription)
    
    if prompt:
        return prompt.decode("utf-8")
    else:
        return None


def load_dataset_with_metadata(image_dir):

    os.makedirs(target_dir, exist_ok=True)

    data = []

    for filename in os.listdir(image_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            source_path = os.path.join(image_dir, filename)
            target_path = os.path.join(target_dir, filename)
            
            # Copy the file to the target directory
            shutil.copy2(source_path, target_path)
            
            # Read the prompt from the original file
            prompt = read_prompt_from_metadata(source_path)
            
            # Add the prompt to the copied file's metadata
            if prompt:
                add_prompt_to_metadata(target_path, prompt)
            
            data.append({
                'filename': filename,
                'prompt': prompt
            })
    
    return Dataset.from_dict(data)