import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
#import albumentations as A
#from albumentations.pytorch import ToTensorV2

import sys
sys.path.insert(0, '../')

from earlyStopping import EarlyStopping
from model import UNET_2d
from model import UNET_2d_noSkip
from better_model import ResUnet2d
from better_model import FullResUnet2d
from tqdm import tqdm
from utils.utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders
)

# HYPERPARAMETERS

LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 100
NUM_WORKERS = 1
TRAIN_DIR_X = '../SeeTrough/gigadose/tryToFit/trainIn'
TRAIN_DIR_Y = '../SeeTrough/gigadose/tryToFit/trainOut'

#TRAIN_DIR_X = '../SeeTrough/undersample/JTS/240_trainIn'
#TRAIN_DIR_Y = '../SeeTrough/undersample/JTS/trainOut'

TRAIN_NAME = "tryToFit_1e3"
DIMENSION = '2d'
MODEL = ResUnet2d(in_channels=1, out_channels=1).to(DEVICE)

VAL_DIR_X = '../SeeTrough/gigadose/tryToFit/testIn'
VAL_DIR_Y = '../SeeTrough/gigadose/tryToFit/testOut' 

#VAL_DIR_X = '../SeeTrough/undersample/JTS/240_testIn'
#VAL_DIR_Y = '../SeeTrough/undersample/JTS/testOut' 

# TRAIN

def train(loader, val_loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader) 

    model.train()

    save_loss = 0 # for loss average calculation
    cont = 0

    # RUNNING TROUGH ALL THE BATCHES
    for batch_idx, (data, targets) in enumerate(loop):
        # print(data.dtype)
        # torch.unsqueeze(data, 1).to(device = DEVICE)
        data = torch.unsqueeze(data, 1).to(device = DEVICE)
        targets = torch.unsqueeze(targets, 1).to(device = DEVICE)
    
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions.float(), targets.float())

        save_loss += loss.item()
        cont += 1

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss = loss.item())
    
    # END OF EPOCH, USE VALIDATE SET TO MONITORIZE OVERFITTING
    model.eval()
    cont_val = 0
    loss_sum = 0
    loop1 = tqdm(val_loader)
    for idx, (x, y) in enumerate(loop1):
        x = torch.unsqueeze(x, 1).to(device = DEVICE)
        y = torch.unsqueeze(y, 1).to(device = DEVICE)
    
        with torch.no_grad():
            preds = model(x)
            vloss = loss_fn(preds.float(), y.float())
            # loop.set_description(f"vloss: {vloss:>7f}, {es.status}")

        cont_val += 1
        loss_sum += vloss.item()


    return save_loss/cont, loss_sum/cont_val


def main():
    
    print(DEVICE)
    print(TRAIN_NAME)
    print(MODEL)
    # This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.
    torch.backends.cudnn.benchmark =  True
    torch.backends.cudnn.enabled =  True

    model = MODEL

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    es = EarlyStopping()

    # train_loader, val_loader = fget_loader...
    train_loader = get_loaders(
        DIMENSION,
        TRAIN_DIR_X,
        TRAIN_DIR_Y,
        # VAL_DIR_X,
        # VAL_DIR_Y,
        BATCH_SIZE,
        # train_transform
    )

    val_loader = get_loaders(
        DIMENSION,
        VAL_DIR_X,
        VAL_DIR_Y,
        BATCH_SIZE
    )

    scaler = torch.cuda.amp.GradScaler()

    save_loss = np.empty(0, dtype=np.float32)
    save_val = np.empty(0, dtype=np.float32)

    for epoch in range(NUM_EPOCHS):
        print(f"epoch: ({epoch})")
        average_loss, val_loss = train(train_loader, val_loader, model, optimizer, loss_fn, scaler)
        save_loss = np.append(save_loss, average_loss)
        save_val = np.append(save_val, val_loss)
        print(average_loss)
        print(val_loss)


    np.save(TRAIN_NAME, save_loss)
    np.save(TRAIN_NAME+"_val", save_val)


    # Save model
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(checkpoint, TRAIN_NAME+".pth.tar")

if __name__ == "__main__":
    main()

