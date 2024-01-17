# prende input [rete] [dataset] [nome_file_output.tif]
# data una rete e un file .tif ritorna in output

import sys
import os
import numpy as np
import pydicom

import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../2d')


from pydicom.dataset import Dataset
from model import UNET_2d
import torch
from utils.utils import (
    load_checkpoint
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = UNET_2d(in_channels=1, out_channels=1).to(DEVICE) 


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
        print("ERR: Usage\npython3 get_volume [net_weights] [dataset] [output_name.dcm]")
        return 1

    load(sys.argv[1]) # load checkpoint in argv[1]

    # creo array per volume di input
    input = sorted(os.listdir(sys.argv[2]))

    # per ogni i in input passo alla rete e salvo output
    output = []
    for i in input:
        image = np.load(os.path.join(sys.argv[2], i))
        # pred = pred_image(image)
        output.append(image)


    megaOutput = np.stack(output)
    # normalize 0-255
    megaOutput = (megaOutput - np.min(megaOutput)) / (np.max(megaOutput) - np.min(megaOutput)) * 255
    
    print(megaOutput.shape[0])

    # Create a new DICOM dataset
    dataset = Dataset()

    # Set required DICOM attributes
    dataset.PatientName = "Tom"
    dataset.PatientID = "123456"
    dataset.Modality = "CT"
    # dataset.SeriesInstanceUID = UID.generate_uid()

    # Set the transfer syntax
    dataset.is_little_endian = True
    dataset.is_implicit_VR = True

    # Set image-related DICOM attributes
    dataset.Rows = megaOutput.shape[1]
    dataset.Columns = megaOutput.shape[2]
    dataset.BitsAllocated = 16
    dataset.SamplesPerPixel = 1
    dataset.NumberOfFrames = megaOutput.shape[0]
    dataset.PixelData = megaOutput.astype(np.uint16).tobytes()

    # Save the DICOM dataset to a file
    filename = sys.argv[3]
    pydicom.filewriter.write_file(filename, dataset)

    return 0 

if __name__ == "__main__":
    main()
