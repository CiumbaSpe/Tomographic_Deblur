import sys
import os
import numpy as np
import pydicom
from pydicom.dataset import Dataset

def main():

    if (len(sys.argv) < 2):
        print("ERR: Usage\n python3 prova.py [file.npy]")

    megaOutput = np.load(sys.argv[1])

    print(megaOutput.shape)

    output = np.empty(0)
        
    # 0,0
    # 0,1 1,0
    # 0,2 1,1 2,0
    output = np.append(output, megaOutput[0,0])
    # output = np.append(output, megaOutput[0,0])
    # ...

    # 0,3 1,2 2,1 3,0
    # 1,3 2,2 3,1 4,0
    # 2,3 3,2 4,1 5,0
    for i in range(megaOutput.shape[0] - 6):
        cont = i
        for j in reversed(range(4)):
            print(cont, j)
            cont += 1
        
        print()

    # 3,3 4,2 5,1
    # 4,3 5,2 
    # 5,3



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
    # dataset.Rows = megaOutput.shape[2]
    # dataset.Columns = megaOutput.shape[3]
    # dataset.BitsAllocated = 16
    # dataset.SamplesPerPixel = 1
    # dataset.NumberOfFrames = megaOutput.shape[0] * megaOutput.shape[1]
    # dataset.PixelData = megaOutput.astype(np.uint16).tobytes()

    # # Save the DICOM dataset to a file
    # filename = "daicom"
    # pydicom.filewriter.write_file(filename, dataset)

    return 0

if __name__ == "__main__":
    main()