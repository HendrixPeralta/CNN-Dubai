# %%

import os 
import cv2 
from PIL import Image 
import numpy as np 
from patchify import patchify
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# %%
minmaxscaler = MinMaxScaler()
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

# %%
image_patches = patchify(image, (patch_size, patch_size, 3), step=patch_size) 
image_patches
# %%
len(image_patches)
# %%
image.shape[0]//patch_size*patch_size
# %%
print(type(image))
# %%
print(type(Image.fromarray(image)))

# %%
image_dataset = []
mask_dataset = []

for image_type in ["images" , "masks"]:
    if image_type == "images":
        image_extension = "jpg" 
    else: 
        image_extension = "png" 
    for tile_id in range(1,8):
        for image_id in range(1,20):
            image = cv2.imread(f"{dataset_root_folder}/{dataset_name}/Tile {tile_id}/{image_type}/image_part_00{image_id}.{image_extension}")
            if image is not None: 
                #print(image.shape)
                size_x = (image.shape[0]//patch_size)*patch_size
                size_y = (image.shape[1]//patch_size)*patch_size
                #print("{} === {} - {}".format(image.shape, size_x, size_y))
                image = Image.fromarray(image)
                image = image.crop((0, 0, size_x, size_y))
                #print("{}, {}".format(image.size[0],image.size[1]))
                
                #Converting back to an array 
                image = np.array(image)
                image_patches = patchify(image, (patch_size, patch_size,3), step=patch_size)
                
                #selecting the images from the patches 
                for i in range(image_patches.shape[0]):
                    for j in range(image_patches.shape[1]):
                        individual_patched_images = image_patches[i,j,:,:]
                        # print(individual_patched_images.shape)
                        
                        # MinMaxing 
                        individual_patched_images = minmaxscaler.fit_transform(individual_patched_images.reshape(-1, individual_patched_images.shape[-1])).reshape(individual_patched_images.shape)
                        
                        individual_patched_images = individual_patched_images[0]
                        # print(individual_patched_images.shape)

                        if image_type == "images":
                            image_dataset.append(individual_patched_images)
                        else: 
                            mask_dataset.append(individual_patched_images)

                    
# %%
print(len(image_dataset))
# %%
print(len(mask_dataset))
# %%
 