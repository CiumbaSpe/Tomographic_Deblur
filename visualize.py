import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import os
import sys
from sklearn.metrics import mean_squared_error
from model import UNET
from utils import (
    load_checkpoint,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "weights/first_seetrough.pth.tar" # default
MODEL = UNET(in_channels=1, out_channels=1).to(DEVICE) 

NUM = 3 # default number of image to show
COLUMN = 3

# test data dir
GTDIR = "./seeTroughDataset/testOut/"
NOISEDIR = "./seeTroughDataset/testIn/"


def pred_image(image, model = MODEL):
        model.eval()
        with torch.no_grad(): # does not calculate gradient
            preds = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(DEVICE)
            preds_tensor = model(preds)

            preds_tensor = preds_tensor.squeeze(0).squeeze(0).cpu()
            return preds_tensor.numpy() 

def normalize(img):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

def visualize_image(num = NUM):
    # pick n images and store in img[]
    img = random.sample(os.listdir(GTDIR), num)

    fig, axs = plt.subplots(num, COLUMN, figsize=(20,10))

    for i in range(len(img)):

        print(GTDIR + img[i])
        print(NOISEDIR + img[i])

        gt = normalize(np.load(GTDIR + img[i]).astype(np.float32))
        noise = normalize(np.load(NOISEDIR + img[i]).astype(np.float32))
        pred = pred_image(noise, MODEL)

        mse_noise = mean_squared_error(gt, noise)
        mse_pred = mean_squared_error(gt, pred)

        axs[i,0].imshow(gt, cmap = 'gray')
        axs[i,0].set_title(img[i])  
        axs[i,1].imshow(noise, cmap = 'gray')
        axs[i,1].set_title("mse: %f" %mse_noise)
        axs[i,2].imshow(pred, cmap = 'gray')
        axs[i,2].set_title("mse: %f" %mse_pred)

        print(f"mse img({img[i]}): ", mse_pred)
        
    for i in range(num):
        for j in range(COLUMN):
            axs[i,j].set_xticks([])
            axs[i,j].set_yticks([])

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
        load(sys.argv[1]) # load checkpoint in argv[1]
    else: 
        load() # load default const CHECKPOINT

    # check argument 'number of images'
    if len(sys.argv) >= 3:
        visualize_image(int(sys.argv[2]))
    else:
        visualize_image() # load default number of images

    return 0

if __name__ == '__main__':
    main()


