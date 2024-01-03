import numpy as np
import torch
import os
import sys
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from model import UNET
from utils import (
    load_checkpoint, normalize
)


CHECKPOINT = "weights/first_seetrough.pth.tar" # default value
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = UNET(in_channels=1, out_channels=1).to(DEVICE) 

# cosa vuoi testare

GTDIR = "./gigadose_dataset/testOut/"
NOISEDIR = "./gigadose_dataset/testIn/"

def pred_image(image, model):
        model.eval()
        with torch.no_grad(): # does not calculate gradient
            preds = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(DEVICE)
            preds_tensor = model(preds)

            preds_tensor = preds_tensor.squeeze(0).squeeze(0).cpu()
            return preds_tensor.numpy()


def test(model):

    mse_corrupted = 0
    mse_pred = 0
    psnr_corrupted = 0
    psnr_pred = 0
    ssim_corrupted = 0
    ssim_pred = 0

    n = len(os.listdir(GTDIR))

    for i in os.listdir(GTDIR):
        gt = normalize(np.load(GTDIR + i).astype(np.float32))
        noise = normalize(np.load(NOISEDIR + i).astype(np.float32))
        
        mse_corrupted += mean_squared_error(gt, noise)
        psnr_corrupted += psnr(gt, noise)
        ssim_corrupted += ssim(gt, noise, data_range=noise.max() - noise.min())

        preds = pred_image(noise, model)

        mse_pred += mean_squared_error(gt, preds)
        psnr_pred += psnr(gt, preds)
        ssim_pred += ssim(gt, preds, data_range=preds.max() - preds.min())



    print("average mse for corrupted images in dataset " + NOISEDIR + "\n", mse_corrupted/n)
    print("average mse for deblurred images\n", mse_pred/n)

    print("average psnr for corrupted images\n", psnr_corrupted/n)
    print("average psnr for deblurred images\n", psnr_pred/n )

    print("average ssim for corrupted images\n", ssim_corrupted/n)
    print("average ssim for deblurred images\n", ssim_pred/n )


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

