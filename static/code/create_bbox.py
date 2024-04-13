"""
The following example code iterates through all the images and creates: 
(i) bounding boxes for both lft test object and test result window in each image
(ii) a bounding box for only lft test object  in each image
(iii) a bounding box for only test result window in each image
The ground-truth bounding boxes can then be used for model training, evaluation, and other tasks.

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

covid_lft_bbox_dict_path = "./lft_grounding/annotations/covid_lft_bbox.json"

covid_lft_bbox = json.load(
    file := open(covid_lft_bbox_dict_path, "r")
); file.close()

images = covid_lft_bbox.keys()

# Get the current working directory
cwd = os.getcwd()

# Define the directory names
base_dir = "bounding_box"

base_dir_path = os.path.join(cwd, base_dir)

# Create the base directory if it doesn't exist
if not os.path.exists(base_dir_path):
    os.mkdir(base_dir_path)
    print(f"Directory '{base_dir}' created at {base_dir_path}.")
 
def get_covid_test_bbox(image):
    
    ann = covid_lft_bbox[image]
    return [ann['Covid Test']]

def get_covid_test_result_bbox(image):
    
    ann = covid_lft_bbox[image]
    return [ann['Test Result']]
 
def get_gt_dimensions(image):
    
    ann = covid_lft[image]
    return ann["Ground Truth Dimensions"]
    
def save_combined_bbox():
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
    
        colors = [
            (200, 0, 250), # Purple for the first bbox
            (255, 165, 0)   # Orange for the second bbox
        ]
    
        test_result = np.array(annotation["Test Result"][0])
        covid_test = np.array(annotation["Covid Test"][0])
       
        bbox_covid_test = get_covid_test_bbox(image)
        bbox_result_window = get_covid_test_result_bbox(image)
        bboxes = [bbox_covid_test,bbox_result_window]
        
        for bbox, color in zip(bboxes, colors):
            cv2.rectangle(resized_image, (bbox[0][0], bbox[0][1]), (bbox[0][2], bbox[0][3]), color, 5)
        
        sub_dir = "combined" 
        sub_dir_path = os.path.join(base_dir_path, sub_dir)
        
        # Create the sub-directory inside the base directory if it doesn't exist
        if not os.path.exists(sub_dir_path):
            os.mkdir(sub_dir_path)
            print(f"Sub-directory '{sub_dir}' created inside '{base_dir}'.")
        else:
            print(f"Sub-directory '{sub_dir}' already exists inside '{base_dir}'.")

        # plt.imshow(resized_image) #uncomment this line for your own testing purposes
        file_path = os.path.join(sub_dir_path, image)
        plt.imsave(file_path, resized_image) 

def save_bbox(bbox_type = "lft_test_object"):
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

        
        sub_dir = ""
        bbox = None
        color = None
        
        if bbox_type == "lft_test_object":
            bbox = get_covid_test_bbox(image)
            color = [200,0,255]
            sub_dir = "lft_test_object" 
        elif bbox_type == "test_result_window":
            bbox = get_covid_test_result_bbox(image)
            color = [255,165,0]
            sub_dir = "test_result_window" 
        
        cv2.rectangle(resized_image, (bbox[0][0], bbox[0][1]), (bbox[0][2], bbox[0][3]), color, 5)
        
        sub_dir_path = os.path.join(base_dir_path, sub_dir)
        
        # Create the sub-directory inside the base directory if it doesn't exist
        if not os.path.exists(sub_dir_path):
            os.mkdir(sub_dir_path)
            print(f"Sub-directory '{sub_dir}' created inside '{base_dir}'.")
        else:
            print(f"Sub-directory '{sub_dir}' already exists inside '{base_dir}'.")

        # plt.imshow(resized_image) #uncomment this line for your own testing purposes
        file_path = os.path.join(sub_dir_path, image)
        plt.imsave(file_path, resized_image)    


    
if __name__=="__main__":
    save_combined_bbox()
    
    #uncomment this line and set bbox_type as "lft_test_object" or "test_result_window" based on your requirements
    # save_bbox()
  
