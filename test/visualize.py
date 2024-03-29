import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import os
import sys
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, '../')
sys.path.insert(0, '../3d')
sys.path.insert(0, '../2d')

from model import UNET_2d
from model import UNET_2d_noSkip
from better_model import ResUnet2d
from better_model import FullResUnet2d
# from model import UNET_3d
from utils.utils import (
    load_checkpoint
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "weights/first_seetrough.pth.tar" # default
MODEL = UNET_2d_noSkip(in_channels=1, out_channels=1).to(DEVICE) 

NUM = 3 # default number of image to show
COLUMN = 3

# test data dir
GTDIR = "../SeeTrough/gigaJS/testOut/"
NOISEDIR = "../SeeTrough/gigaJS/testIn/"

# GTDIR = "../SeeTrough/undersample/testOut/"
# NOISEDIR = "../SeeTrough/undersample/120_testIn/"

def pred_image(image, model = MODEL):
        model.eval()
        with torch.no_grad(): # does not calculate gradient
            preds = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(DEVICE)
            preds_tensor = model(preds)

            preds_tensor = preds_tensor.squeeze(0).squeeze(0).cpu()
            return preds_tensor.numpy() 


def visualize_image(num = NUM):
    # pick n images and store in img[]
    #img = random.sample(os.listdir(GTDIR), num)

    a = sorted(os.listdir(GTDIR))
    img = []

    if(num == 1):
        img.append(a[300])
    if(num == 2):
        img.append(a[300])
        img.append(a[500])

    fig, axs = plt.subplots(num, COLUMN, figsize=(20,10))

    for i in range(len(img)):

        print(GTDIR + img[i])
        print(NOISEDIR + img[i])

        gt = np.load(GTDIR + img[i])
        noise = np.load(NOISEDIR + img[i])

        pred = pred_image(noise, MODEL)
        # pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred)) * 255
        print(np.min(pred))
        print(np.max(pred))
                                        
        #mse_noise = mean_squared_error(gt, noise)
        #mse_pred = mean_squared_error(gt, pred)

        ssim_noise = ssim(gt, noise, data_range=noise.max() - noise.min())
        ssim_pred = ssim(gt, pred, data_range=pred.max() - pred.min())

        if(len(img) != 1):
          axs[i,0].imshow(gt, cmap = 'gray')
          axs[i,0].set_title(img[i])  
          axs[i,1].imshow(noise, cmap = 'gray')
          axs[i,1].set_title("ssim: %f" %ssim_noise)
          axs[i,2].imshow(pred, cmap = 'gray')
          axs[i,2].set_title("ssim: %f" %ssim_pred)
        else:
          axs[0].imshow(gt, cmap = 'gray')
          axs[0].set_title(img[i])  
          axs[1].imshow(noise, cmap = 'gray')
          axs[1].set_title("ssim: %f" %ssim_noise)
          axs[2].imshow(pred, cmap = 'gray', vmin=0, vmax=255)
          axs[2].set_title("ssim: %f" %ssim_pred)
        

        print(f"ssim img({img[i]}): ", ssim_pred)
        
    for i in range(num):
        for j in range(COLUMN):
            if(len(img) != 1):
              axs[i,j].set_xticks([])
              axs[i,j].set_yticks([])
            else:
              axs[j].set_xticks([])
              axs[j].set_yticks([])


    plt.savefig("risultati.png")

    plt.show()        

# load the model and the weights
def load(checkpoint = CHECKPOINT):
    print()
    if DEVICE == "cuda":
        load_checkpoint(torch.load(checkpoint), MODEL)
    else:
        load_checkpoint(torch.load(checkpoint, map_location=torch.device('cpu')), MODEL)



def main():
    
    # check argument 'checkpoint'
    if len(sys.argv) >= 2:
        print("loading: ", sys.argv[1])
        load(sys.argv[1]) # load checkpoint in argv[1]
    else: 
        print("loading: ", CHECKPOINT)
        load() # load default const CHECKPOINT

    # check argument 'number of images'
    if len(sys.argv) >= 3:
        visualize_image(int(sys.argv[2]))
    else:
        visualize_image() # load default number of images

    return 0

if __name__ == '__main__':
    main()


