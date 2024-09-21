import os
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image, ImageCms
import numpy as np

# Resize and convert images to sRGB IEC61966-2.1
def format_and_save_img(save_path, source_img_path, img_dim):
    with Image.open(source_img_path) as img:
        img_resized = img.resize((img_dim, img_dim))

        # Convert to sRGB IEC61966-2.1
        srgb_profile = ImageCms.createProfile("sRGB")
        icc_profile = img_resized.info.get('icc_profile')
        if icc_profile:
            try:
                img_resized = ImageCms.profileToProfile(img_resized, icc_profile, srgb_profile, outputMode='RGB')
            except (TypeError, ImageCms.PyCMSError) as e:
                print(f"Warning: Failed to convert ICC profile for {source_img_path}. Error: {e}. Converting to RGB.")
                img_resized = img_resized.convert('RGB')
        else:
            img_resized = img_resized.convert('RGB')

        img_resized.save(save_path, icc_profile=img.info.get('icc_profile'))  # Save the resized image back to the new resized directory


def format_data(dataset_folder, dataset_name, img_dim):
    # Create a dataframe to store the pairs of text descriptions and image paths
    data = {'text_description': [], 'image_path': []}
    dataset_path = os.path.join(dataset_folder, dataset_name)

    resized_directory = os.path.join(dataset_folder, f"{dataset_name}_{img_dim}")
    if not os.path.exists(resized_directory):
        os.makedirs(resized_directory)

    # Iterate through each flower category
    # for category, index in os.listdir(dataset_path):
    for index, category in enumerate(os.listdir(dataset_path)):
        element = os.path.join(dataset_path, category)

        new_image_name = f"planet-{index}.jpg"
        resized_element_path = os.path.join(resized_directory, new_image_name)

        # Check if it's a directory
        if os.path.isdir(element):
            # Iterate through images in the category
            for image_name in os.listdir(element):
                image_path = os.path.join(element, image_name)
                format_and_save_img(resized_element_path, image_path, img_dim)
                data['text_description'].append(f"planet")
                data['image_path'].append(resized_element_path)
        else:
            format_and_save_img(resized_element_path, element, img_dim)
            data['text_description'].append(f"planet")
            data['image_path'].append(resized_element_path)
            
    if len(data) < 2:
        raise Exception("Not sufficient data found in the specified directory")

    # Create a dataframe from the data
    df = pd.DataFrame(data)

    train_df = df
    # Split the dataset into training and testing sets
    # train_df, test_df = train_test_split(df, test_size=0., random_state=42)

    # Save the dataframes to CSV files
    train_df_path = os.path.join(dataset_folder, 'train_dataset.csv')
    test_df_path = os.path.join(dataset_folder, 'test_dataset.csv')
    train_df.to_csv(train_df_path, index=False)
    # test_df.to_csv(test_df_path, index=False)

    # return train_df, test_df
    return train_df


def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)

    # Normalize the image to [-1, 1]
    img_array = (img_array / 255.0) * 2 - 1

    # print('Image shape:', img_array.shape)
    # print('Image :', img_array)

    return img_array