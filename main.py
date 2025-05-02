# %%

import os 
import cv2 
from PIL import Image 
import numpy as np 
from patchify import patchify
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib import pyplot as plt
import random

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
    elif image_type == "masks": 
        image_extension = "png" 
    for tile_id in range(1,8):
        for image_id in range(1,20):
            image = cv2.imread(f"{dataset_root_folder}/{dataset_name}/Tile {tile_id}/{image_type}/image_part_00{image_id}.{image_extension}")
            if image is not None: 
                if image_type == "masks":
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
                        if image_type == "images":
                            individual_patched_images = image_patches[i,j,:,:]
                        # print(individual_patched_images.shape)
                         
                        # MinMaxing 
                            individual_patched_images = minmaxscaler.fit_transform(individual_patched_images.reshape(-1, individual_patched_images.shape[-1])).reshape(individual_patched_images.shape)
                            individual_patched_images = individual_patched_images[0]
                            # print(individual_patched_images.shape)
                            image_dataset.append(individual_patched_images)
                        elif image_type == "masks": 
                            individual_patched_masks = image_patches[i,j,:,:]
                            individual_patched_masks = individual_patched_masks[0] 
                            mask_dataset.append(individual_patched_masks)

                    
# %%
print(len(image_dataset))
print(len(mask_dataset))
# %%
mask_dataset[0]
# %%
image_dataset = np.array(image_dataset)
mask_dataset = np.array(mask_dataset) 
# %%
print(len(image_dataset))
print(len(mask_dataset))

# %%
random_image_id = random.randint(0, len(image_dataset))

plt.figure(figsize=(14,8))

plt.subplot(121)  # First subplot
plt.imshow(image_dataset[random_image_id])
plt.title("Image Patch")
plt.axis("off")

plt.subplot(122)  # Second subplot
plt.imshow(mask_dataset[random_image_id], cmap='gray')  # If mask is grayscale
plt.title("Mask Patch")
plt.axis("off")

plt.tight_layout()
plt.show()

 
# %%
plt.imshow(mask_dataset[0])
# %%
# Labels definition 

    # Building: #3C1098
    # Land (unpaved area): #8429F6
    # Road: #6EC1E4
    # Vegetation: #FEDD3A
    # Water: #E2A929
    # Unlabeled: #9B9B9B
 
class_building = "#3C1098"
class_building = class_building.strip("#")
class_building = np.array(tuple(int(class_building[i:i+2],16)for i in (0,2,4)))
print(class_building)

class_land = "#8429F6"
class_land = class_land.strip("#")
class_land = np.array(tuple(int(class_land[i:i+2],16)for i in (0,2,4)))
print(class_land)

class_road = "#6EC1E4"
class_road = class_road.strip("#")
class_road = np.array(tuple(int(class_road[i:i+2],16)for i in (0,2,4)))
print(class_road)

class_vegetation = "#FEDD3A"
class_vegetation = class_vegetation.strip("#")
class_vegetation = np.array(tuple(int(class_vegetation[i:i+2],16)for i in (0,2,4)))
print(class_vegetation)

class_water = "#E2A929"
class_water = class_water.strip("#")
class_water = np.array(tuple(int(class_water[i:i+2],16)for i in (0,2,4)))
print(class_water)

class_unlabeled = "#9B9B9B"
class_unlabeled = class_unlabeled.strip("#")
class_unlabeled = np.array(tuple(int(class_unlabeled[i:i+2],16)for i in (0,2,4)))
print(class_unlabeled)


# %%
mask_dataset.shape[0]
# %%
# 
def rgb_to_label(label):
    label_segment = np.zeros(label.shape, dtype=np.uint8)
    label_segment[np.all(label == class_water, axis=-1)] = 0 
    label_segment[np.all(label == class_land, axis=-1)] = 1
    label_segment[np.all(label == class_road, axis=-1)] =2
    label_segment[np.all(label == class_building, axis=-1)] = 3
    label_segment[np.all(label == class_vegetation, axis=-1)] = 4
    label_segment[np.all(label == class_unlabeled, axis=-1)] = 5
    return label_segment

# %%
labels = []
for i in range(mask_dataset.shape[0]):
    label = rgb_to_label(mask_dataset[i])
    labels.append(label)
    
labels = np.array(labels)
# %%
print(len(labels))
# %%
labels[450]
# %%
