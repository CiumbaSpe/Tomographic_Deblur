import numpy as np
import torch
import torch.nn as nn
import os
import sys
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm

sys.path.insert(0, '../')
sys.path.insert(0, '../3d')
sys.path.insert(0, '../2d')

from model import UNET_2d
from model import UNET_2d_noSkip
from better_model import ResUnet2d
from better_model import FullResUnet2d
# from model import UNET_3d
from utils.utils import (
    load_checkpoint, get_loaders
)

sys.path.insert(0, '../')
sys.path.insert(0, '../3d')
sys.path.insert(0, '../2d')


CHECKPOINT = "weights/first_seetrough.pth.tar" # default value
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = ResUnet2d(in_channels=1, out_channels=1).to(DEVICE) 
BATCH_SIZE = 4
DIMENSION = '2d'

# cosa vuoi testare
GTDIR = "../SeeTrough/gigadose/JTS/testOut/"
NOISEDIR = "../SeeTrough/gigadose/JTS/testIn/"

#GTDIR = "../SeeTrough/undersample/JTS/testOut/"
#NOISEDIR = "../SeeTrough/undersample/JTS/240_testIn/"

# def pred_image(image, model):
#         model.eval()
#         with torch.no_grad(): # does not calculate gradient
#             preds = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(DEVICE)
#             preds_tensor = model(preds)

#             preds_tensor = preds_tensor.squeeze(0).squeeze(0).cpu()
#             return preds_tensor.numpy()

#def pred_image(data, model = MODEL):
#        model.eval()
#        with torch.no_grad(): # does not calculate gradient
#            preds_tensor = model(data)
#            preds_tensor = preds_tensor.squeeze(0).squeeze(0).cpu()
#            return preds_tensor.numpy() 


def test(model):

    mse_corrupted = 0
    mse_pred = 0
    psnr_corrupted = 0
    psnr_pred = 0
    ssim_corrupted = 0
    ssim_pred = 0

    loader = get_loaders(
        DIMENSION,
        NOISEDIR,
        GTDIR,
        BATCH_SIZE,
    )

    loop = tqdm(loader) 

    loss_fn = nn.MSELoss()
    cont = 0
    loss_sum = 0
    # RUNNING TROUGH ALL THE BATCHES
    MODEL.eval()
    for idx, (x, y) in enumerate(loop):
        x = torch.unsqueeze(x, 1).to(device = DEVICE)
        y = torch.unsqueeze(y, 1).to(device = DEVICE)
    
        with torch.no_grad():
            preds = model(x)
            vloss = loss_fn(preds.float(), y.float())
            # loop.set_description(f"vloss: {vloss:>7f}, {es.status}")

        cont += 1
        loss_sum += vloss.item()

    # n = len(os.listdir(GTDIR))
    
    # cont = 0
    # for i in sorted(os.listdir(GTDIR)):
    #     if (cont >= 21 or cont <= 839):
    #         gt = np.load(GTDIR + i)
    #         noise = np.load(NOISEDIR + i)
        
    #         mse_corrupted += mean_squared_error(gt, noise)
    #         psnr_corrupted += psnr(gt, noise)
    #         ssim_corrupted += ssim(gt, noise, data_range=noise.max() - noise.min())

    #         preds = pred_image(noise, model)

    #         mse_pred += mean_squared_error(gt, preds)
    #         psnr_pred += psnr(gt, preds)
    #         ssim_pred += ssim(gt, preds, data_range=preds.max() - preds.min())
    #     cont = cont + 1


    # print("average mse for corrupted images in dataset " + NOISEDIR + "\n", mse_corrupted/n)
    # print("average mse for deblurred images\n", mse_pred/n)

    # print("\naverage psnr for corrupted images\n", psnr_corrupted/n)
    # print("average psnr for deblurred images\n", psnr_pred/n )

    # print("\naverage ssim for corrupted images\n", ssim_corrupted/n)
    # print("average ssim for deblurred images\n", ssim_pred/n )
        
    print("mse: ", loss_sum/cont)


# load the model and the weights
def load(checkpoint = CHECKPOINT):
    if DEVICE == "cuda":
        load_checkpoint(torch.load(checkpoint), MODEL)
    else:
        load_checkpoint(torch.load(checkpoint, map_location=torch.device('cpu')), MODEL)


def main():
    if len(sys.argv) >= 2:
        print("loading: ", sys.argv[1])
        load(sys.argv[1])
    else: 
        print("loading: ", CHECKPOINT)
        load() # load default const CHECKPOINT

    print("testing:")
    test(MODEL)

    return 0

if __name__ == '__main__':
    main()

