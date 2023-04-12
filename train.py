import os
import wandb
from infovae import InfoVAE
from img2dataset import download
import torch
from torch.utils.data import DataLoader, Dataset
import glob
import cv2
import os
from PIL import Image
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple
from dataset import IMDBFaces

train_dataset = IMDBFaces(data_path='/content/faces/content/faces', split='train', transform=data_transform)
test_dataset = IMDBFaces(data_path='/content/faces/content/faces', split='test', transform=Rescale(32))

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)


wandb.init(
      # Set the project where this run will be logged
      project="InfoVAE",  
      # Track hyperparameters and run metadata
      config={
      "learning_rate": LR,
      "architecture": "InfoVAE",
      "dataset": "IMDB_faces",
      "epochs": EPOCHS,
      "augmentation": 'random flip',
      "latent_dim": LATENT_DIM
      })
CHECKPOINTS_DIR = 'checkpnts'
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
model = InfoVAE(latent_dim=LATENT_DIM)


optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
model.to(device)
criterion = nn.MSELoss()


for epoch in range(EPOCHS):
    for i, data in enumerate(train_dataloader):
        imgs = data
        imgs = imgs.to(device)
        optimizer.zero_grad()
        model.train()

        out, mu, logVar = model(imgs)
        loss = criterion(out, imgs)
   
        # loss = train_loss['loss']
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(f'{epoch+1}/{EPOCHS} loss: {loss.item()}')
            wandb.log({"loss": loss, "mu": mu, "logVar": logVar})


    scheduler.step()
    torch.save(model.state_dict(), f'{CHECKPOINTS_DIR}/infoVAE_{epoch}.pth' )
