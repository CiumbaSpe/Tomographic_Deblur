# prende input [rete] [dataset] [nome_file_output.dcm]

import sys
import os
import numpy as np
import tqdm

import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../3d')

import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import UID, ExplicitVRLittleEndian

from model_3d import UNET_3d
from better_model_3d import ResUnet3d
from tqdm import tqdm
import torch
from utils.utils import (
    load_checkpoint, get_loaders
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = ResUnet3d(in_channels=1, out_channels=1).to(DEVICE) 
BATCH_SIZE = 1
DIMENSION = '3d'

def pred_image(data, model = MODEL):
        model.eval()
        with torch.no_grad(): # does not calculate gradient
            preds_tensor = model(data)
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

    loader = get_loaders(
        DIMENSION,
        sys.argv[2],
        sys.argv[2],
        BATCH_SIZE,
        shuffle=False
    )

    loop = tqdm(loader) 

    cont = 0

    # RUNNING TROUGH ALL THE BATCHES
    MODEL.eval()
    for batch_idx, (data, targets) in enumerate(loop):
        # print(idx)
        # if(idx != False):
        data = torch.unsqueeze(data, 1).to(device = DEVICE)
        if(data.shape[2] == 4): # should prevent downsizing to 0
            pred = pred_image(data)
            # pred = (pred - np.min(pred)) / (np.max(pred) - np.min(pred)) * 255
            output.append(pred)
            # cont += 1
            # if cont == 10:
            #     break

    megaOutput = np.stack(output)

    # CALCOLO LA MEDIA CON STRIDE 1
    output = []    
    l = megaOutput.shape[0]
    # 0,0
    # 0,1 1,0
    # 0,2 1,1 2,0
    output.append(megaOutput[0,0])
    output.append(np.mean([megaOutput[0,1], megaOutput[1,0]], axis=0))
    output.append(np.mean([megaOutput[0,2], megaOutput[1,1], megaOutput[2,0]], axis = 0))

    # 0,3 1,2 2,1 3,0
    # 1,3 2,2 3,1 4,0
    # 2,3 3,2 4,1 5,0
    # ...
    for i in range(megaOutput.shape[0] - 6):
        output.append(np.mean([megaOutput[i,3], megaOutput[i+1, 2], megaOutput[i+2,1], megaOutput[i+3,0]], axis = 0))
        
    output.append(np.mean([megaOutput[l-3, 3], megaOutput[l-2, 2], megaOutput[l-1, 1]], axis = 0))    
    output.append(np.mean([megaOutput[l-2, 3], megaOutput[l-1, 2]], axis = 0))
    output.append(megaOutput[l-1, 3])
    # 3,3 4,2 5,1
    # 4,3 5,2 
    # 5,3

    megaOutput = np.stack(output)

    # normalize 0-255
    megaOutput = (megaOutput - np.min(megaOutput)) / (np.max(megaOutput) - np.min(megaOutput)) * 4095
    megaOutput = np.resize(megaOutput, (920, 836, 836))

    megaOutput = megaOutput.astype(np.uint16)
    print(megaOutput.shape)
    print(np.min(megaOutput))
    print(np.max(megaOutput))

    # Create a new DICOM dataset
    dataset = Dataset()

    # Set required DICOM attributes
    dataset.PatientName = "Tom"
    dataset.PatientID = "123456"
    dataset.Modality = "CT"

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = UID("1.2.840.10008.5.1.4.1.1.2")
    file_meta.MediaStorageSOPInstanceUID = UID("1.2.3")
    file_meta.ImplementationClassUID = UID("1.2.3.4")
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    # Add the file meta information
    dataset.file_meta = file_meta

    # Set the transfer syntax
    dataset.is_little_endian = True
    dataset.is_implicit_VR = True

    # Set image-related DICOM attributes
    dataset.PixelSpacing = [0.12, 0.12]
    # dataset.PixelSpacing = 0.12
    dataset.Rows = megaOutput.shape[1]
    dataset.Columns = megaOutput.shape[2]
    dataset.BitsAllocated = 16
    dataset.SamplesPerPixel = 1
    dataset.NumberOfFrames = megaOutput.shape[0] 

    dataset.PixelRepresentation = 0
    dataset.HighBit = 15
    dataset.BitsStored = 16
    # dataset.SmallestImagePixelValue = np.min(megaOutput)
    # dataset.LargestImagePixelValue = np.max(megaOutput)
    dataset.PixelData = megaOutput.tobytes()

 
    # Save the DICOM dataset to a file
    filename = sys.argv[3]
    # dataset.PixelData = pixel_array.tostring()

    # dataset.save_as('gio.dcm')

    pydicom.filewriter.dcmwrite(filename, dataset)

    return 0 

if __name__ == "__main__":
    main()
