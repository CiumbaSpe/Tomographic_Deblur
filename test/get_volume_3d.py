# prende input [rete] [dataset] [nome_file_output.tif]
# data una rete e un file .tif ritorna in output

import sys
import os
import numpy as np
import pydicom
import tqdm

import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../3d')

from pydicom.dataset import Dataset
from model_3d import UNET_3d
import torch
from utils.utils import (
    load_checkpoint, normalize, get_loaders
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = UNET_3d(in_channels=1, out_channels=1).to(DEVICE) 
SLICES = 4

def pred_image(image, model = MODEL):
        model.eval()
        with torch.no_grad(): # does not calculate gradient
            preds = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(DEVICE)
            preds_tensor = model(preds)

            preds_tensor = preds_tensor.squeeze(0).squeeze(0).cpu()
            return preds_tensor.numpy() 

# load the model and the weights
def load(checkpoint):
    print()
    if DEVICE == "cuda":
        load_checkpoint(torch.load(checkpoint), MODEL)
    else:
        load_checkpoint(torch.load(checkpoint, map_location=torch.device('cpu')), MODEL)



def main():

    if (len(sys.argv) < 4):
        print("ERR: Usage\npython3 get_volume [net_weights] [dataset] [output_name]")
        return 1

    load(sys.argv[1]) # load checkpoint in argv[1]

    # creo array per volume di input
    input = sorted(os.listdir(sys.argv[2]))

    # per ogni i in input passo alla rete e salvo output
    output = []
    stack_slices = []

    loader = get_loaders(
        sys.argv[1],
        sys.argv[1],
        1,
    )

    loop = tqdm(loader) 

    MODEL.train()
    # RUNNING TROUGH ALL THE BATCHES
    for batch_idx, (data, targets) in enumerate(loop):
        # print(data.dtype)
        # torch.unsqueeze(data, 1).to(device = DEVICE)
        data = torch.unsqueeze(data, 1).to(device = DEVICE)
        print(data.shape)

    # for i in range(len(input)):
    #     for j in range(SLICES):
    #         image = normalize(np.load(os.path.join(sys.argv[2], input[i + j])).astype(np.float32))
    #         stack_slices.append(image)

    #     image_3d = np.stack(stack_slices)

    #     pred = pred_image(image_3d)
    #     pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred)) * 255
    #     #output.append(pred)

    megaOutput = np.stack(output)
    print(megaOutput.shape)

    # # Create a new DICOM dataset
    # dataset = Dataset()

    # # Set required DICOM attributes
    # dataset.PatientName = "Tom"
    # dataset.PatientID = "123456"
    # dataset.Modality = "CT"
    # # dataset.SeriesInstanceUID = UID.generate_uid()

    # # Set the transfer syntax
    # dataset.is_little_endian = True
    # dataset.is_implicit_VR = True

    # # Set image-related DICOM attributes
    # dataset.Rows = megaOutput.shape[1]
    # dataset.Columns = megaOutput.shape[2]
    # dataset.BitsAllocated = 16
    # dataset.SamplesPerPixel = 1
    # dataset.NumberOfFrames = megaOutput.shape[0]
    # dataset.PixelData = megaOutput.astype(np.uint16).tobytes()

    # # Save the DICOM dataset to a file
    # filename = sys.argv[3]
    # pydicom.filewriter.write_file(filename, dataset)

    return 0 

if __name__ == "__main__":
    main()