"""
The following example code iterates through all the images and creates: 
(i) binary ground truth masks for both lft test object and test result window in each image
(ii) a binary ground truth mask for only lft test object  in each image
(iii) a binary ground truth mask for only test result window in each image
The ground-truth masks can then be used for model training, evaluation, and other tasks.

Ensure to update the path as needed. This code assumes the JSON files and image directories are 
stored at the root level of the project directory. 
"""
import cv2
import json
import imutils
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

covid_lft_dict_path = "./lft_grounding/annotations/covid_lft.json"

covid_lft = json.load(
    file := open(covid_lft_dict_path, "r")
); file.close()

images = covid_lft.keys()

# Get the current working directory
cwd = os.getcwd()

# Define the directory names
base_dir = "binary_masks"

base_dir_path = os.path.join(cwd, base_dir)

# Create the base directory if it doesn't exist
if not os.path.exists(base_dir_path):
    os.mkdir(base_dir_path)
    print(f"Directory '{base_dir}' created at {base_dir_path}.")
    
def save_combined_masks():
    for image in images:
        annotation = covid_lft[image]
        url = "https://bivpriv.colorado.edu/covid_lft/" + image
        gt_dimensions = annotation["Ground Truth Dimensions"]
        gt_width = gt_dimensions[0]
        gt_height = gt_dimensions[1]
        
        original_image = imutils.url_to_image(url)
        original_dimensions = original_image.shape
        original_height = int(original_dimensions[0])
        original_width = int(original_dimensions[1])
    
        scale_width = gt_width / float(original_width)
        scale_height = gt_height / float(original_height)
        scale = min(scale_width, scale_height)
    
        resized_image = imutils.resize(
            original_image, 
            width=int(original_width * scale), 
            height=int(original_height * scale)
        )
    
        resized_dimensions = resized_image.shape
        resized_height = int(resized_dimensions[0])
        resized_width = int(resized_dimensions[1])
    
        stencil = np.zeros(
            resized_image.shape, 
            dtype=np.uint8
        ).astype(resized_image.dtype)
        
        colors = [
            (200, 0, 250), # Purple for the first polygon
            (255, 165, 0)   # Orange for the second polygon
        ]
    
        test_result = np.array(annotation["Test Result"][0])
        covid_test = np.array(annotation["Covid Test"][0])
        polygons = [covid_test, test_result]
        
        for polygon, color in zip(polygons, colors):
            mask = cv2.fillPoly(stencil, [polygon], color)
        

        sub_dir = "combined" 
        sub_dir_path = os.path.join(base_dir_path, sub_dir)
        
        # Create the sub-directory inside the base directory if it doesn't exist
        if not os.path.exists(sub_dir_path):
            os.mkdir(sub_dir_path)
            print(f"Sub-directory '{sub_dir}' created inside '{base_dir}'.")
        else:
            print(f"Sub-directory '{sub_dir}' already exists inside '{base_dir}'.")

        # plt.imshow(stencil) #uncomment this line for your own testing purposes
        file_path = os.path.join(sub_dir_path, image)
        plt.imsave(file_path, stencil) 

def save_mask(mask_type = "lft_test_object"):
    for image in images:
        annotation = covid_lft[image]
        url = "https://bivpriv.colorado.edu/covid_lft/" + image
        gt_dimensions = annotation["Ground Truth Dimensions"]
        gt_width = gt_dimensions[0]
        gt_height = gt_dimensions[1]
        
        original_image = imutils.url_to_image(url)
        original_dimensions = original_image.shape
        original_height = int(original_dimensions[0])
        original_width = int(original_dimensions[1])
    
        scale_width = gt_width / float(original_width)
        scale_height = gt_height / float(original_height)
        scale = min(scale_width, scale_height)
    
        resized_image = imutils.resize(
            original_image, 
            width=int(original_width * scale), 
            height=int(original_height * scale)
        )
    
        resized_dimensions = resized_image.shape
        resized_height = int(resized_dimensions[0])
        resized_width = int(resized_dimensions[1])
    
        stencil = np.zeros(
            resized_image.shape, 
            dtype=np.uint8
        ).astype(resized_image.dtype)
        
        color = [255, 255, 255]
    
        test_result = np.array(annotation["Test Result"][0])
        covid_test = np.array(annotation["Covid Test"][0])
        
        sub_dir = ""
        
        if mask_type == "lft_test_object":
            mask = cv2.fillPoly(stencil, [covid_test], color)
            sub_dir = "lft_test_object" 
        elif mask_type == "test_result_window":
            mask = cv2.fillPoly(stencil, [test_result], color)
            sub_dir = "test_result_window" 

        sub_dir_path = os.path.join(base_dir_path, sub_dir)
        
        # Create the sub-directory inside the base directory if it doesn't exist
        if not os.path.exists(sub_dir_path):
            os.mkdir(sub_dir_path)
            print(f"Sub-directory '{sub_dir}' created inside '{base_dir}'.")
        else:
            print(f"Sub-directory '{sub_dir}' already exists inside '{base_dir}'.")

        # plt.imshow(stencil) #uncomment this line for your own testing purposes
        file_path = os.path.join(sub_dir_path, image)
        plt.imsave(file_path, stencil)    

# use this function if you wish to overlay both masks onto the image
#if you wish to save the image, set save to True
def overlay_masks_on_image(image, save=False):
    annotation = covid_lft[image]
    url = "https://bivpriv.colorado.edu/covid_lft/" + image
    gt_dimensions = annotation["Ground Truth Dimensions"]
    gt_width = gt_dimensions[0]
    gt_height = gt_dimensions[1]
    
    original_image = imutils.url_to_image(url)
    original_dimensions = original_image.shape
    original_height = int(original_dimensions[0])
    original_width = int(original_dimensions[1])

    scale_width = gt_width / float(original_width)
    scale_height = gt_height / float(original_height)
    scale = min(scale_width, scale_height)

    resized_image = imutils.resize(
        original_image, 
        width=int(original_width * scale), 
        height=int(original_height * scale)
    )
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    resized_dimensions = resized_image.shape
    resized_height = int(resized_dimensions[0])
    resized_width = int(resized_dimensions[1])

    covid_test_color = [255, 0, 100] # Cyan for Covid Test
    test_result_color = [0, 250, 0] # Yellow for Test Result


    stencil_covid_test = np.zeros(
    resized_image.shape, 
    dtype=np.uint8
    ).astype(resized_image.dtype)

    stencil_test_result = np.zeros(
    resized_image.shape, 
    dtype=np.uint8
    ).astype(resized_image.dtype)

    test_result = np.array(annotation["Test Result"][0])
    covid_test = np.array(annotation["Covid Test"][0])
    polygons = [covid_test, test_result]
    
    covid_test_mask = cv2.fillPoly(stencil_covid_test, [covid_test], covid_test_color)
    test_result_mask = cv2.fillPoly(stencil_test_result, [test_result], test_result_color)
    # Combine the masks with the original image
    result_covid_test = cv2.bitwise_or(resized_image, stencil_covid_test)
    result_test_result = cv2.bitwise_or(resized_image, stencil_test_result)

    # Combine the results to show both masks
    result = cv2.bitwise_or(result_covid_test, result_test_result)

    plt.imshow(result)
    
    if save:
        sub_dir = "overlayed_masks" 
        sub_dir_path = os.path.join(base_dir_path, sub_dir)
        
        # Create the sub-directory inside the base directory if it doesn't exist
        if not os.path.exists(sub_dir_path):
            os.mkdir(sub_dir_path)
            print(f"Sub-directory '{sub_dir}' created inside '{base_dir}'.")
        else:
            print(f"Sub-directory '{sub_dir}' already exists inside '{base_dir}'.")

        # plt.imshow(stencil) #uncomment this line for your own testing purposes
        file_path = os.path.join(sub_dir_path, image)
        plt.imsave(file_path, stencil)
    
    
if __name__=="__main__":
    save_combined_masks()
    
    #uncomment this line and set mask_type as "lft_test_object" or "test_result_window" based on your requirements
    # save_mask()
  
    #set save as "True" to save the generated image
    #set desired image name
    #uncomment the following lines to overlay both masks onto an image 
    #example_image_name = "Covid-Test-Positive-62P.jpg"
    # overlay_masks_on_image(example_image_name,True)