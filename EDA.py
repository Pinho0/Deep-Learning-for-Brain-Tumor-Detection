"""
EDA for Brain Tumor Classification Dataset

This script performs exploratory data analysis (EDA) on a brain tumor classification dataset,
including image visualization, dataset balance, and computing normalization statistics.

"""

# ---------------------
# 1. Library Imports
# ---------------------

import numpy as np
import pandas as pd
import os
import PIL
import random
from tqdm import tqdm 
from collections import Counter 
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader

# ---------------------
# 2. Hardware Setup
# ---------------------

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using {device} device.")

# ---------------------
# 3. Dataset Paths
# ---------------------

train_dir = '\dataset\Training'
test_dir = '\dataset\Testing'
classes = os.listdir(train_dir)
print(classes)

# ---------------------
# 4. Visualize Random Images from Each Class
# ---------------------

def sample_images(data_path, classname): 
    class_dir = os.path.join(data_path, classname) 
    if not os.path.exists(class_dir):
        return "Invalid directory"
    image_list = os.listdir(class_dir) 
    if len(image_list) < 4:
        return "Not enough images in folder"
    images_sample = random.sample(image_list, 4) 
    # Plot 
    plt.figure(figsize=(20, 20))
    for i in range(4):
        img_loc = os.path.join(class_dir, images_sample[i])
        img = PIL.Image.open(img_loc)
        plt.subplot(1, 4, i + 1)
        plt.imshow(img)
        plt.axis("off")
    plt.show()

# 3 examples of each type of brain tumor and the helthy one
sample_images(train_dir, "glioma")
sample_images(train_dir, "meningioma")
sample_images(train_dir, "pituitary")
sample_images(train_dir, "notumor")

# ---------------------
# 5. Dataset Size & Image Properties
# ---------------------

glioma_dir = os.path.join(train_dir,"glioma")
print(f"Number of glioma image: {len(os.listdir(glioma_dir))}")
meningioma_dir = os.path.join(train_dir,"meningioma")
print(f"Number of meningioma image: {len(os.listdir(meningioma_dir))}")
notumor_dir = os.path.join(train_dir,"notumor")
print(f"Number of notumor image: {len(os.listdir(notumor_dir))}")
pituitary_dir = os.path.join(train_dir,"pituitary")
print(f"Number of pituitary image: {len(os.listdir(pituitary_dir))}")

meningioma_sample = os.listdir(meningioma_dir)[0] 
meningioma_sample_img = PIL.Image.open(os.path.join(meningioma_dir, meningioma_sample)) 
glioma_sample = os.listdir(glioma_dir)[0]
glioma_sample_img = PIL.Image.open(os.path.join(glioma_dir, glioma_sample))
notumor_sample = os.listdir(notumor_dir)[0]
notumor_sample_img = PIL.Image.open(os.path.join(notumor_dir, notumor_sample))
pituitary_sample = os.listdir(pituitary_dir)[0]
pituitary_sample_img = PIL.Image.open(os.path.join(pituitary_dir, pituitary_sample))

print(meningioma_sample_img.mode, meningioma_sample_img.size)
print(glioma_sample_img.mode, glioma_sample_img.size)
print(notumor_sample_img.mode, notumor_sample_img.size)
print(pituitary_sample_img.mode, pituitary_sample_img.size)
print('the size of the iamgens is also diferente')

# Normalizing the dataset
class ConvertImage:
    def __call__(self, img):
        if img.mode != 'RGB':
            img = img.convert("RGB")
        return img    
    
# ---------------------
# 6. Initial Image Transform (Pre-normalization)
# ---------------------

transform_nonorm = transforms.Compose( 
    [
        ConvertImage(), 
        transforms.Resize((224, 224)), 
        transforms.ToTensor() 
    ]
)

batch_size=32
dataset = datasets.ImageFolder(train_dir, transform=transform_nonorm) 
loader = DataLoader(dataset, batch_size=batch_size) 

# ---------------------
# 7. Compute Mean and Standard Deviation
# ---------------------

def mean_std(loader):
    count, count_square, num_batches = 0, 0, 0 
    for data, _ in tqdm(loader):
        count += torch.mean(data, dim=[0, 2, 3]) 
        count_square += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1 
        
    mean = count / num_batches 
    std = (count_square / num_batches - mean ** 2) ** 0.5 
    
    return mean, std

mean, std = mean_std(loader) 
print(mean)
print(std)

# ---------------------
# 8. Final Transforms and Dataset Setup
# ---------------------
     
# Until now have only done a "standarzition" of the datasets and elments of the datasets, now we normalize them.  
transform = transforms.Compose(
    [
        ConvertImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

batch_size = 32
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

def class_counts(dataset):
    c = Counter(x[1] for x in tqdm(dataset))
    try: 
        class_to_index = dataset.class_to_idx 
    except AttributeError: 
        class_to_index = dataset.dataset.class_to_idx 
    return pd.Series({cat: c[idx] for cat, idx in class_to_index.items()}) 


train_counts = class_counts(train_dataset)
train_counts.plot(kind="bar")
plt.xlabel("Class Label")
plt.ylabel("Frequency [count]")
plt.title("Distribution of Classes in Training Dataset")
plt.show()

print(train_counts)

val_counts = class_counts(test_dataset)
val_counts.plot(kind="bar")
plt.xlabel("Class Label")
plt.ylabel("Frequency [count]")
plt.title("Distribution of Classes in Validation Dataset")
plt.show()

