#-------------------------------------------------------------------------------
# convert_data.py
# Author: Thi Ngoc Nhi Nguyen
# Date: 10-12-2025
#------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image
import glob
import shutil
import cv2

# ------------------------------------------------------------------------------
#  CONFIGURATION
# ------------------------------------------------------------------------------

# Setup paths
ORIGINAL_FOLDER = 'data1/fashion_mnist'
CLEANED_FOLDER = 'data1/new_fashion_mnist'
OUTPUT_FILE = 'data1/fashion_mnist_dataset.npz'
IMAGES_PATH = glob.glob(f'{ORIGINAL_FOLDER}/*.png')
IMAGES_NEW_PATH = glob.glob(f'{CLEANED_FOLDER}/*.png')

# Variables
NUM_CLASSES = 10
TARGET_SAMPLES = 6000
UNDERSAMPLED_LABEL = 5 # because the labels 5 just have 200 images


# Create the folder trash_dir if not exist
if not os.path.exists(CLEANED_FOLDER):
    os.makedirs(CLEANED_FOLDER)
      
def clean_images():
    """ Step 1: Clean images:  Check Quality of images. If image is white (pixel =255), then fix the images
            Copy the good images and fixed the bad images and save it in the output_path
    """

    for filename in IMAGES_PATH:
        # Load the image and convert to grayscale ('L' mode)
        # This converts RGB (3 channels) to a single channel (0-255) for processing 
        print("Step 1:  Load dataset")
        img = Image.open(filename)   #-> data1\fashion_mnist\1-12337.png
        
        print("Step 2: Check Quality of images")
        img_array = np.array(img)
         
        # Calculate the mean pixel intensity for each row and column
        row_means = np.mean(img_array, axis=1)
        col_means = np.mean(img_array, axis=0)

        # Check if any row or column is entirely white (mean =255)
        is_good = np.all(row_means < 255) and np.all(col_means < 255)
        mask = (img_array == 255).astype(np.uint8) * 255 # boolen -> astype 0 or 1 *255 -> 0 0 255 0 255
        
        if is_good:
            # if good image -> Copy to CLEANED_FOLDER
            file_basename = os.path.basename(filename)
            shutil.copy(filename, CLEANED_FOLDER)
            print(f"Good image: {file_basename}")
        else:
            # if bad image -> delete the row or column and save it again
            file_basename = os.path.basename(filename)
            output_path = os.path.join(CLEANED_FOLDER, file_basename)
            print("Fixing the bad image: ", filename)
            # CV2 Inpaint
            img_array = cv2.inpaint(img_array, mask, 3, cv2.INPAINT_TELEA) # get around 3 pixel to calculate with math inpainting 
            # Convert to PIL Image
            fixed_img = Image.fromarray(img_array.astype(np.uint8)) # cv2.inpaint return float or int32 , PIL just get image in unit8
            print("Image Fixed!", file_basename)
            fixed_img.save(output_path)
   
def extract_label_from_filename(filename):
    """Get label of each image: '1-12337.png' -> 1"""
    parts = filename.split('-', 1)  #1-12337.png -> ['1', '12337.png']
    return int(parts[0])  # ->1
    
#-------------------------------------------------------------------------------
# 4.1 Extract the label from the filename in npz
# 1-12337.png  -> Label 1 
#              -> Imagename: 12337.png
#-------------------------------------------------------------------------------
def extract_and_save_npz():
    """_summary_
    """
    labels =[]
    images=[]
    for filename in os.listdir(ORIGINAL_FOLDER):
        
        # Open Image
        file_path  = os.path.join(ORIGINAL_FOLDER, filename)
        img = Image.open(file_path )
        img_array = np.array(img)
        
        # get labels from filename
        label = extract_label_from_filename(filename) #1-12337.png -> ['1', '12337.png']

        images.append(img_array)
        labels.append(int(label)) # make sure return in int not float
        
    # covert in array
    images = np.array(images)
    labels = np.array(labels)
    
    # save in npz
    np.savez_compressed(OUTPUT_FILE, labels=labels, images=images)

#-------------------------------------------------------------------------------
# 3.1 Check the distribution of each class. If there is an imbalance between the classes in the dataset
# ->duplicate the images of the underrepresented classes to balance the dataset.
#-------------------------------------------------------------------------------

def print_distribution(labels):
    """Print the number of images for each label """
    for i in range(NUM_CLASSES):
        count = np.sum(labels == i)
        print(f"  Label {i}: {count} images") # -> 0-9_ 6000 Image but 5: 200 Images 
        
def balance_dataset():
    """_summary_

    Returns:
        _type_: _description_
    """
    dataset = np.load(OUTPUT_FILE)
    labels = dataset["labels"]   
    images = dataset["images"]
    labels_oversampling = 5
    
    # print("\n=== count amount image of each label ===")
    print_distribution(labels)  # -> 0-9_ 6000 Image but 5: 200 Images  
    
    # Oversampling at Undersampled label ( label 5)
    label_5_indices = np.where(labels == UNDERSAMPLED_LABEL)[0] # labels type (54200,) ->[0]
    label_5_images = images[label_5_indices]
    
    # Random sample to choose more 5800 images for label 5
    need_to_add_acount = TARGET_SAMPLES-len(label_5_images) # 5800 for label 5
    additional_indices = np.random.choice(len(label_5_images), 
                                          size=need_to_add_acount, 
                                          replace=True)  # One image of labels can be choice more time
    
    additional_images = label_5_images[additional_indices]
    additional_labels = np.ones(need_to_add_acount, dtype=labels.dtype)*labels_oversampling # if dont use dtype=labels.dtype -> return float 5. but labels has type int64
    
    # Merge old images and added images
    images = np.concatenate([images, additional_images], axis=0) # because type image (54200,28,28)
    labels = np.concatenate([labels, additional_labels], axis=0)
    
    # check again
    print_distribution(labels)
    
    
#-------------------------------------------------------------------------------
# 4. Visualization the dataset
#-------------------------------------------------------------------------------
def visualization():
    """_summary_
    """
    data = np.load(OUTPUT_FILE) 
    images = data['images']
    labels = data['labels']
    print(labels.shape)
    # indices_label_3 = np.where(labels == 3) -> (array([12000, 12001, 12002, ..., 17997, 17998, 17999], shape=(6000,)),)
    indices_label_3 = np.where(labels == 3)[0]  # -> get array
    indices_label_5 = np.where(labels == 5)[0]
    print(indices_label_3)


    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        
    axes[0].imshow(images[indices_label_3[0]], cmap='gray')
    axes[0].set_title(f'Image: {indices_label_3[0]}')

    axes[1].imshow(images[indices_label_5[0]], cmap='gray')
    axes[1].set_title(f'Image. {indices_label_5[0]}')

    plt.show()  

#-------------------------------------------------------------------------------
# run
#-------------------------------------------------------------------------------
def run():
    # Step 1: Clean images
    clean_images()
    
    # Step 2: Extract and save to npz
    extract_and_save_npz()
    
    # Step 4. Extract and save to npz
    balance_dataset()
    
       
#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------
if __name__ == "__main__":
    run()
