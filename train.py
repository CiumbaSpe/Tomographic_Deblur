import torch
import torch.nn as nn
import torch.optim as optim
#import albumentations as A
#from albumentations.pytorch import ToTensorV2
from earlyStopping import EarlyStopping
from model import UNET
from tqdm import tqdm
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders
)

# HYPERPARAMETERS

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
NUM_EPOCHS = 20
NUM_WORKERS = 1
TRAIN_DIR_X = 'undersample_dataset/120_trainIn'
TRAIN_DIR_Y = 'undersample_dataset/trainOut'
# VAL_DIR_X = 'new_mayo/FBPB/mayo_val/'
# VAL_DIR_Y = 'new_mayo/GT/mayo_val/' 

# TRAIN

def train(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader) 
    # steps = list(enumerate(loader))

    model.train()

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

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss = loss.item())
    
    
    return False

def main():
    
    print(DEVICE)
    # This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.
    torch.backends.cudnn.benchmark =  True
    torch.backends.cudnn.enabled =  True

    model = UNET(in_channels=1, out_channels=1).to(DEVICE)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    es = EarlyStopping()

    # train_loader, val_loader = fget_loader...
    train_loader = get_loaders(
        TRAIN_DIR_X,
        TRAIN_DIR_Y,
        # VAL_DIR_X,
        # VAL_DIR_Y,
        BATCH_SIZE,
        # train_transform
    )

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        print(f"epoch: ({epoch})")
        if(train(train_loader, model, optimizer, loss_fn, scaler)):
            break

    # Save model
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(checkpoint, "120_first_train.pth.tar")

if __name__ == "__main__":
    main()

