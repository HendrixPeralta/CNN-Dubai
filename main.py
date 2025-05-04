# %%

import os 
import cv2 
import numpy as np 
import random

from patchify import patchify
from matplotlib import pyplot as plt
from PIL import Image 

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.metrics import MeanIoU
from sklearn.model_selection import train_test_split 
from tensorflow.keras.utils import to_categorical

scaler = MinMaxScaler()
# %%
dataset_root_folder = r"C:\Users\hendr\Desktop\programming\CNN-Dubai\resources"
dataset_name = "Semantic segmentation dataset"
patch_size = 256
# %%

# %%
image_dataset = []
for path, subdirs, files in os.walk(os.path.join(dataset_root_folder,dataset_name)):
    dir_name = path.split(os.path.sep)[-1]
    if dir_name == "images": 
        images = os.listdir(path)
        #print(path)
        for i, image_name in enumerate(images):
            if (image_name.endswith(".jpg")):
                image = cv2.imread(path+"/"+image_name,1)
                SIZE_X = (image.shape[1]//patch_size)*patch_size
                SIZE_Y = (image.shape[0]//patch_size)*patch_size
                image = Image.fromarray(image)
                image = image.crop((0, 0, SIZE_X, SIZE_Y))            
                
                image = np.array(image)
                
                # Extract Patches 
                patched_img = patchify(image, (patch_size,patch_size, 3), step=patch_size)
                
                for i in range(patched_img.shape[0]):
                    for j in range(patched_img.shape[1]):
                        single_patched_img = patched_img[i,j,:,:]
                        
                        single_patched_img = scaler.fit_transform(single_patched_img.reshape(-1,single_patched_img.shape[-1])).reshape(single_patched_img.shape)
                        single_patched_img = single_patched_img[0]
                        image_dataset.append(single_patched_img)
  # %%
mask_dataset = []
for path, subdirs, files in os.walk(os.path.join(dataset_root_folder,dataset_name)):
    dir_name = path.split(os.path.sep)[-1]
    if dir_name == "masks":
        mask = os.listdir(path)
        for i, mask_name in enumerate(mask):
            if (mask_name.endswith(".png")):
                mask = cv2.imread(path+"/"+mask_name,1)
                mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
                SIZE_X = (mask.shape[1]//patch_size)*patch_size
                SIZE_Y = (mask.shape[0]//patch_size)*patch_size
                mask =Image.fromarray(mask)
                mask = mask.crop((0, 0, SIZE_X, SIZE_Y))
                
                mask = np.array(mask)
                
                patched_mask = patchify(mask, (patch_size, patch_size,3), step=patch_size)
                
                for i in range(patched_mask.shape[0]):
                    for j in range(patched_mask.shape[1]):
                        single_patched_mask = patched_mask[i, j, :, :]
                        
                        # single_patched_mask = scaler.fit_transform(single_patched_mask.reshape(-1,single_patched_mask.shape[-1])).reshape(single_patched_mask.shape)
                        
                        single_patched_mask = single_patched_mask[0]
                        mask_dataset.append(single_patched_mask)
                        
                        
                        
# %%
image_dataset = np.array(image_dataset)
mask_dataset = np.array(mask_dataset)                        
# %%
image_id = random.randint(0, len(image_dataset))
plt.figure(figsize=(12,6))
plt.subplot(121)
plt.imshow(np.reshape(image_dataset[image_id], (patch_size,patch_size,3)))
plt.subplot(122)
plt.imshow(np.reshape(mask_dataset[image_id], (patch_size, patch_size, 3)))
plt.show()


# random_image_id = random.randint(0, len(image_dataset))

# %%
# Labels definition 

    # Building: #3C1098
    # Land (unpaved area): #8429F6
    # Road: #6EC1E4
    # Vegetation: #FEDD3A
    # Water: #E2A929
    # Unlabeled: #9B9B9B
 
building = "#3C1098"
building = building.strip("#")
building = np.array(tuple(int(building[i:i+2],16)for i in (0,2,4)))
print(building)

land = "#8429F6"
land = land.strip("#")
land = np.array(tuple(int(land[i:i+2],16)for i in (0,2,4)))
print(land)

road = "#6EC1E4"
road = road.strip("#")
road = np.array(tuple(int(road[i:i+2],16)for i in (0,2,4)))
print(road)

vegetation = "#FEDD3A"
vegetation = vegetation.strip("#")
vegetation = np.array(tuple(int(vegetation[i:i+2],16)for i in (0,2,4)))
print(vegetation)

water = "#E2A929"
water = water.strip("#")
water = np.array(tuple(int(water[i:i+2],16)for i in (0,2,4)))
print(water)

unlabeled = "#9B9B9B"
unlabeled = unlabeled.strip("#")
unlabeled = np.array(tuple(int(unlabeled[i:i+2],16)for i in (0,2,4)))
print(unlabeled)


# %%
# 
def rgb_to_label(label):
    label_segment = np.zeros(label.shape, dtype=np.uint8)
    label_segment[np.all(label == building, axis=-1)] = 0
    label_segment[np.all(label == land, axis=-1)] = 1
    label_segment[np.all(label == road, axis=-1)] =2
    label_segment[np.all(label == vegetation, axis=-1)] = 3
    label_segment[np.all(label == water, axis=-1)] = 4 
    label_segment[np.all(label == unlabeled, axis=-1)] = 5
    
    label_segment = label_segment[:,:,0]
    return label_segment

# %%
labels = []
for i in range(mask_dataset.shape[0]):
    label = rgb_to_label(mask_dataset[i])
    labels.append(label)
    
    
labels = np.array(labels)
labels = np.expand_dims(labels, axis=3)
# %%
print(np.unique(labels))
# %%

image_id = random.randint(0, len(image_dataset))

plt.figure(figsize=(14,8))

plt.subplot(121)  # First subplot
plt.imshow(image_dataset[image_id])
plt.title("Image Patch")
plt.axis("off")

plt.subplot(122)  # Second subplot
plt.imshow(mask_dataset[image_id])  # If mask is grayscale
plt.title("Mask Patch")
plt.axis("off")

plt.tight_layout()
plt.show()
# %%
n_classes = len(np.unique(labels))
labels_cat = to_categorical(labels, num_classes=n_classes)

# %%
# Creating training data 
X_train, x_test, Y_train, y_test = train_test_split(image_dataset, labels_cat, test_size=0.20, random_state=15)

training_test_datasets = [X_train, x_test, Y_train, y_test]

for dataset in training_test_datasets:
    print(dataset.shape)
    
# ------------------^---------------------------------------------------------
# %%
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.layers import concatenate, BatchNormalization, Dropout, Lambda
# %%
from keras import backend as k 
# %%
def jaccard_coef(y_true, y_pred):
    y_pred_flatten = k.flatten(y_pred)
    y_true_flatten = k.flatten(y_true)
    intersection = k.sum(y_true_flatten * y_pred_flatten)
    iou = (intersection + 1.0 ) / (k.sum(y_true_flatten) + k.sum(y_pred_flatten) - intersection + 1.0)
    return iou
# %%
# Bulding the model 
def unet_model(n_classes, image_height, image_width, image_channels):
    input = Input((image_height, image_width, image_channels))
    c1 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(input)
    c1 = Dropout(0.2)(c1)
    c1 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c1)
    p1 = MaxPooling2D((2,2))(c1)
    
    c2 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p1)
    c2 = Dropout(0.2)(c2)
    c2 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c2)
    p2 = MaxPooling2D((2,2))(c2)
    
    c3 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c3)
    p3 = MaxPooling2D((2,2))(c3)
    
    c4 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c4)
    p4 = MaxPooling2D((2,2))(c4)
    
    c5 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(p4)
    c5  = Dropout(0.2)(c5)
    c5 = Conv2D(256, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c5)
    
    #Expansive path
    
    u6 = Conv2DTranspose(128, (2,2), strides=(2,2), padding="same")(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c6)
    
    u7 = Conv2DTranspose(64, (2,2), strides=(2,2), padding="same")(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c7)
    
    u8 = Conv2DTranspose(32, (2,2), strides=(2,2), padding="same")(c7)
    u8 = (concatenate[u8, c2])
    c8 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(32, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c8)
    
    u9 = Conv2DTranspose(16, (2,2), strides=(2,2), padding="same")(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(u9)
    c9 = Dropout(0.2)(c9)
    c9 = Conv2D(16, (3,3), activation="relu", kernel_initializer="he_normal", padding="same")(c9)
    
    output = Conv2D(n_classes, (1,1), activation="softmax")(c9)
    model = Model(inputs=[input], outputs=output)
    
    return model

# %%
metrics = ["accuracy", jaccard_coef]
# %%

image_height = X_train.shape[1]
image_width = X_train.shape[2]
image_channels = X_train.shape[3]

def get_model():
    return unet_model(n_classes=n_classes, image_height=image_height, image_width=image_width, image_channels=image_channels)
# %%
model = get_model()
# %%
