import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from model import UNET
from utils import (
    load_checkpoint,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "big_long_train_B16.pth.tar"

# load the numpy file
data1 = np.load('./new_mayo/FBPB/mayo_test/C016206.npy')
data2 = np.load('./new_mayo/GT/mayo_test/C016206.npy')
data3 = np.load('./new_mayo/FBPB/mayo_test/C016206.npy')

data4 = np.load('./new_mayo/FBPB/mayo_test/C030128.npy')
data5 = np.load('./new_mayo/GT/mayo_test/C030128.npy')
data6 = np.load('./new_mayo/FBPB/mayo_test/C030128.npy')

data7 = np.load('./new_mayo/FBPB/mayo_test/C07732.npy')
data8 = np.load('./new_mayo/GT/mayo_test/C07732.npy')
data9 = np.load('./new_mayo/FBPB/mayo_test/C07732.npy')

# load the net weight
model = UNET(in_channels=1, out_channels=1).to(DEVICE)

if DEVICE == "cuda":
    load_checkpoint(torch.load(CHECKPOINT), model)
else:
    load_checkpoint(torch.load(CHECKPOINT, map_location=torch.device('cpu')), model)

# input = torch.from_numpy(data3).unsqueeze(0).unsqueeze(0).float().to("cuda")
# input = torch.unsqueeze(torch.from_numpy(data3), 0).unsqueeze(0).cpu().to("cuda")

# print(input3.dtype)
# print(input3.shape)
# model.eval()

def pred_image(image):
    with torch.no_grad(): # does not calculate gradient
        preds = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(DEVICE)
        preds_tensor = model(preds)

        #if(DEVICE == "cpu"):
            #preds_tensor = preds_tensor.squeeze(0).squeeze(0)
        #else:
        preds_tensor = preds_tensor.squeeze(0).squeeze(0).cpu()
        return preds_tensor.numpy()

eval_image1 = pred_image(data3)
eval_image2 = pred_image(data6)
eval_image3 = pred_image(data9)


print("mse1: ", mean_squared_error(data2, eval_image1))
print("mse2: ", mean_squared_error(data5, eval_image2))
print("mse3: ", mean_squared_error(data8, eval_image3))


# display the images
ROW = 3
COLUMN = 3 

fig, axs = plt.subplots(ROW, COLUMN, figsize=(20,10))
  
# print("OOOH", mean_squared_error(data2, data1))

axs[0,0].imshow(data2, cmap = 'gray')
axs[0,0].set_title("GT/C004/1.pny")  
axs[0,1].imshow(data1, cmap = 'gray')
axs[0,1].set_title("mse: %f" %mean_squared_error(data2, data1))
axs[0,2].imshow(eval_image1, cmap = 'gray')
axs[0,2].set_title("mse: %f" %mean_squared_error(data2, eval_image1))


axs[1,0].imshow(data5, cmap = 'gray')
axs[1,0].set_title("GT/C016/1.pny")
axs[1,1].imshow(data4, cmap = 'gray')
axs[1,1].set_title("mse: %f" %mean_squared_error(data5, data4))
axs[1,2].imshow(eval_image2, cmap = 'gray')
axs[1,2].set_title("mse: %f" %mean_squared_error(data5, eval_image2))

axs[2,0].imshow(data8, cmap = 'gray')
axs[2,0].set_title("GT/C050/7.pny")
axs[2,1].imshow(data7, cmap = 'gray')
axs[2,1].set_title("mse: %f" %mean_squared_error(data8, data7))
axs[2,2].imshow(eval_image3, cmap = 'gray')
axs[2,2].set_title("mse: %f" %mean_squared_error(data8, eval_image3))

for i in range(ROW):
    for j in range(COLUMN):
        axs[i,j].set_xticks([])
        axs[i,j].set_yticks([])

plt.savefig("risultati.png")

plt.show()

