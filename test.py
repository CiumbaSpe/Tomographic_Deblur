import numpy as np
import torch
import os
import sys
from sklearn.metrics import mean_squared_error
from model import UNET
from utils import (
    load_checkpoint,
)


CHECKPOINT = "weights/first_seetrough.pth.tar" # default value
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = UNET(in_channels=1, out_channels=1).to(DEVICE) 

# cosa vuoi testare

GTDIR = "./seeTroughDataset/testOut/"
NOISEDIR = "./seeTroughDataset/testIn/"

def pred_image(image, model):
        model.eval()
        with torch.no_grad(): # does not calculate gradient
            preds = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(DEVICE)
            preds_tensor = model(preds)

            preds_tensor = preds_tensor.squeeze(0).squeeze(0).cpu()
            return preds_tensor.numpy()


def normalize(img): 
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def test(model):

    mse_corrupted = 0
    mse_pred = 0

    n = len(os.listdir(GTDIR))

    for i in os.listdir(GTDIR):
        gt = normalize(np.load(GTDIR + i).astype(np.float32))
        noise = normalize(np.load(NOISEDIR + i).astype(np.float32))
        
        mse_corrupted += mean_squared_error(gt, noise)

        preds = pred_image(noise, model)

        mse_pred += mean_squared_error(gt, preds)



    print("average mse for corrupted images in dataset " + NOISEDIR + "\n", mse_corrupted/n)
    print("average mse for deblurred images" + "\n", mse_pred/n)

# load the model and the weights
def load(checkpoint = CHECKPOINT):
    if DEVICE == "cuda":
        load_checkpoint(torch.load(checkpoint), MODEL)
    else:
        load_checkpoint(torch.load(checkpoint, map_location=torch.device('cpu')), MODEL)


def main():
    if len(sys.argv) >= 2:
        load(sys.argv[1])
    else: 
        load() # load default const CHECKPOINT

    print("testing")
    test(MODEL)

    return 0

if __name__ == '__main__':
    main()

