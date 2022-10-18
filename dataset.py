import os
import cv2
from tqdm import tqdm
import numpy as np


def get_dataset(images_folder: str, gray=False):
    images, labels = [], []
    folders = os.listdir(images_folder)
    for label in tqdm(folders, total=len(folders)):
        img_folder = f'{images_folder}/{label}'
        for image in os.listdir(img_folder):
            img = cv2.imread(f'{img_folder}/{image}')
            if gray:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)
