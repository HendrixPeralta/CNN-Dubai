# %%

import os 
import cv2 
from PIL import Image 
import numpy as np 
# %%
dataset_root_folder = r"C:\Users\hendr\Desktop\programming\CNN-Dubai\resources"
# %%
dataset_name = "Semantic segmentation dataset"
# %%
for path, subdirs, files in os.walk(os.path.join(dataset_root_folder,dataset_name)):
    dir_name = path.split(os.path.sep)[-1]
    if dir_name == "images": 
        images = os.listdir(path)
        #print(path)
        for i, image_name in enumerate(images):
            if (image_name.endswith(".jpg")):
                print(image_name)        
# %%
image = cv2.imread(f"{dataset_root_folder}/{dataset_name}/Tile 2/images/image_part_001.jpg",1)
 
# %%
patch_size = 256
image.shape[0]//patch_size*patch_size
# %%
print(type(image))
# %%
print(type(Image.fromarray(image)))

# %%
image_dataset = []

image_type = "images" # "masks"
image_extension = "jpg" # "PNG"
for tile_id in range(1,8):
    for image_id in range(1,20):
        image = cv2.imread(f"{dataset_root_folder}/{dataset_name}/Tile {tile_id}/{image_type}/image_part_00{image_id}.{image_extension}")
        if image is not None: 
            #print(image.shape)
            size_x = (image.shape[0]//patch_size)*patch_size
            size_y = (image.shape[1]//patch_size)*patch_size
            print("{} === {} - {}".format(image.shape, size_x, size_y))
            image = Image.fromarray(image)
            image = image.crop((0, 0, size_x, size_y))
            print("{}, {}".format(image.size[0],image.size[1]))
            
# %%
image_dataset
# %%
