# prende input [rete] [volume_input] [nome_file_output]

import sys
import os
import numpy as np
import tifffile
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import UID, ExplicitVRLittleEndian
from better_model_3d import ResUnet3d
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = ResUnet3d(in_channels=1, out_channels=1).to(DEVICE) 
BATCH_SIZE = 1
DIMENSION = '3d'

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

# def pred_image(data, model = MODEL):
#         model.eval()
#         with torch.no_grad(): # does not calculate gradient
#             preds_tensor = model(data)
#             preds_tensor = preds_tensor.squeeze(0).squeeze(0).cpu()
#             return preds_tensor.numpy() 

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
        print("ERR: Usage\npython3 get_volume [net_weights] [input_volume] [output_name]")
        return 1

    load(sys.argv[1]) # load checkpoint in argv[1]


    # controllo se il file in input e' un tiff o un dicom
    postfix = sys.argv[2].split('.')[-1]
    if(postfix == "dcm"):
        dicom_data = pydicom.dcmread(sys.argv[2])
        numpy_array = dicom_data.pixel_array
    elif (postfix == "tif" or postfix == "tiff"):
        numpy_array = tifffile.imread(sys.argv[2])
    else:
        print("ERR: not a dcm or tif file")
        return 2
    
    # normalizzo 0-1 e converto in float32 per la rete
    min = np.min(numpy_array)
    max = np.max(numpy_array)
    numpy_array = (numpy_array - min) / (max - min)
    numpy_array = numpy_array.astype(np.float32) 

    # per ogni input passo alla rete e salvo output
    output = []

    for i in range(numpy_array.shape[0]):
        if(i + 4 < numpy_array.shape[0]):
                pred = pred_image(numpy_array[i:i+4, :, :])
                output.append(pred)

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

    # normalize 0-4095
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
    filename = sys.argv[3].split(".")[0] + ".dcm"
    
    # dataset.PixelData = pixel_array.tostring()

    # dataset.save_as('gio.dcm')

    pydicom.filewriter.dcmwrite(filename, dataset)

    return 0 

if __name__ == "__main__":
    main()
