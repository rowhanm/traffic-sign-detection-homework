from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable

from tqdm import tqdm
import os
import PIL.Image as Image

device = torch.device('cuda')

def get_data(size):
    t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11 = get_transforms(size)
    train_loader = torch.utils.data.DataLoader(
    torch.utils.data.ConcatDataset([
        datasets.ImageFolder('images/train_images', transform=t0),
        datasets.ImageFolder('images/train_images', transform=t1),
        datasets.ImageFolder('images/train_images', transform=t2),
        datasets.ImageFolder('images/train_images', transform=t3),
        datasets.ImageFolder('images/train_images', transform=t4),
        datasets.ImageFolder('images/train_images', transform=t5),
        datasets.ImageFolder('images/train_images', transform=t6),
        datasets.ImageFolder('images/train_images', transform=t7),
        datasets.ImageFolder('images/train_images', transform=t8),
        datasets.ImageFolder('images/train_images', transform=t9),
        datasets.ImageFolder('images/train_images', transform=t10),
    ]),
    batch_size=1024, shuffle=True, num_workers = 8)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('images/val_images', transform=t0),
        batch_size=1024, shuffle=False, num_workers = 8)
    
    return train_loader, val_loader    
    
    
# Get all the possible transforms for a given size.
def get_transforms(size):
    # Keep the same
    t0 = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
    ])

    # Scale brightness between the range (1.5,3.5)
    t1 = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ColorJitter(brightness=2.5),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
    ])

    # Scale saturation between (1,2)
    t2 = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ColorJitter(saturation=2),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
    ])

    # Scale contrast between (1,1.5)
    t3 = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ColorJitter(contrast=1.5),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
    ])

    # Scale hue
    t4 = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ColorJitter(hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
    ])

    # Random horizontal flips
    t5 = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
    ])

    # Random shearing
    t6 = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomAffine(degrees=20, shear=3),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
    ])

    # Random Translation
    t7 = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomAffine(degrees=10, translate=(0.2,0.2)),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
    ])

    # Random perspective change
    t8 = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomPerspective(),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
    ])

    # Random rotation
    t9 = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
    ])

    # Upscale the image to size*1.5 then make a random crop of size=size
    t10 = transforms.Compose([
        transforms.Resize((int(size*1.5), int(size*1.5))),
        transforms.RandomResizedCrop(size=size),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))
    ])

    # TenCrop, only used in TTA
    t11 = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.TenCrop(size),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([transforms.Normalize((0.3337, 0.3064, 0.3171), ( 0.2672, 0.2564, 0.2629))(crop) for crop in crops])),
    ])
    
    return t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11