import numpy as np
import torch
import os
from sklearn.metrics import mean_squared_error
from model import UNET
from utils import (
    load_checkpoint,
)

# load the net weight
model = UNET(in_channels=1, out_channels=1).to("cuda")
load_checkpoint(torch.load("big_long_train_B16.pth.tar"), model)

# cosa vuoi testare
DATA = "mayo_val"

GT_dir = "./new_mayo/GT/" + DATA
FBPB_dir = "./new_mayo/FBPB/" + DATA

mse_corrupted = 0
mse_pred = 0

n = len(os.listdir(GT_dir))

def pred_image(image):
    model.eval()
    with torch.no_grad(): # does not calculate gradient
        preds = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to("cuda")
        preds_tensor = model(preds)

        preds_tensor = preds_tensor.squeeze(0).squeeze(0).cpu()
        return preds_tensor.numpy()


for i, j in zip(os.listdir(GT_dir), os.listdir(FBPB_dir)):
    ground_truth = np.load(os.path.join(GT_dir, i))
    corrupt = np.load(os.path.join(FBPB_dir, j))

    mse_corrupted += mean_squared_error(ground_truth, corrupt)

    preds = pred_image(corrupt)

    mse_pred += mean_squared_error(ground_truth, preds)


print("average mse for corrupted in dataset " + DATA + " = ", mse_corrupted/n)
print("average mse for reconstruct in dataset " + DATA + " = ", mse_pred/n)



