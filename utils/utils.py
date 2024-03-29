import torch
from dataset import SeeTrough2d
from dataset import SeeTrough3d
from torch.utils.data import DataLoader
import numpy as np

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
 
def normalize(img):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = np.nan_to_num(img, 0)
    return img

def get_loaders(
    dimension, # for 2d or 3d dataset
    train_x,
    train_y,
    # val_x,
    # val_y,
    batch_size,
    # train_transform,
    # val_transform,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
):
    if(dimension == '2D' or dimension == '2d'):
        train_ds = SeeTrough2d(
            img_dir_x=train_x,
            img_dir_y=train_y,
            # transform=train_transform,
        )
    elif(dimension == '3D' or dimension == '3d'):
        train_ds = SeeTrough3d(
            img_dir_x=train_x,
            img_dir_y=train_y,
            # transform=train_transform,
        )
    print("shuffle: ", shuffle)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle, 
    )

    # val_ds = MayoDataset(
    #     img_dir_x=val_x,
    #     img_dir_y=val_y,
    #     # transform=val_transform,
    # )

    # val_loader = DataLoader(
    #     val_ds,
    #     batch_size=1,
    #     num_workers=num_workers,
    #     pin_memory=pin_memory,
    #     shuffle=False,
    # )

    # should rerturn val_loader
    return train_loader
