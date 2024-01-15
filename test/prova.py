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
    dataset.Rows = megaOutput.shape[2]
    dataset.Columns = megaOutput.shape[3]
    dataset.BitsAllocated = 16
    dataset.SamplesPerPixel = 1
    dataset.NumberOfFrames = megaOutput.shape[0] * megaOutput.shape[1]
    dataset.PixelData = megaOutput.astype(np.uint16).tobytes()

    # Save the DICOM dataset to a file
    filename = "daicom"
    pydicom.filewriter.write_file(filename, dataset)

    return 0

if __name__ == "__main__":
    main()