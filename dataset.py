import os
import cv2
from tqdm import tqdm
import numpy as np
from skimage import io, transform
from PIL import Image
import json
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import glob

def get_age_label(birthday):
    age = 2023 - birthday
    if age >= 60:
      return 'senior'
    elif age >= 40 and age < 60:
      return 'middle'
    elif age >= 20 and age < 40:
      return 'young'
    elif age >= 13 and age < 20:
      return 'teen'
    return 'kid'

def get_gender_label(sex):
    if sex == 'M':
        return 'male'
    return 'female'

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = np.array(sample)

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w)).astype(np.float32)
        return img

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)

def smart_resize(input_image, new_size):
    input_image = Image.fromarray(input_image)
    width = input_image.width
    height = input_image.height

# Image is portrait or square
    if height >= width:
        crop_box = (0, (height-width)//2, width, (height-width)//2 + width)
        return input_image.resize(size = (new_size,new_size),
                                  box = crop_box)

# Image is landscape
    if width > height:
        crop_box = ((width-height)//2, 0, (width-height)//2 + height, height)
        
        return input_image.resize(size = (new_size,new_size),
                                  box = crop_box)

class IMDBFaces(Dataset):
    """
    URL = https://github.com/marianna13/IMDB_faces
    """
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 ext: str = 'jpg',
                 transform = None
                 ):
        self.data_dir = data_path
        imgs = sorted(glob.glob(f'{data_path}/**/*.{ext}', recursive=True))
        self.transform = transform
        
        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]
        # self.faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):

        img = self.imgs[idx]
        js = img.replace('.jpg', '.json')
        with open(js, 'r') as f:
            d = json.load(f)
            label = get_gender_label(d['SEX'])
        img = Image.fromarray(io.imread(img))
        if self.transform:
            img = self.transform(img)
        return img, label

data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(brightness=.5, hue=.3),
        Rescale(32),
        ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
