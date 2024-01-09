import pydicom
from pydicom.dataset import Dataset
import numpy as np

# Create a new DICOM dataset
dataset = Dataset()

# Set required DICOM attributes
dataset.PatientName = "John_Doe"
dataset.PatientID = "123456"
dataset.Modality = "CT"
# dataset.SeriesInstanceUID = UID.generate_uid()

# Set the transfer syntax
dataset.is_little_endian = True
dataset.is_implicit_VR = True

# Set image-specific attributes
rows = 512
columns = 512
bits_allocated = 16
samples_per_pixel = 1


# Create a NumPy array with synthetic pixel data (replace with your actual image data)
pixel_array = np.random.randint(0, 4096, size=(100, rows, columns), dtype=np.uint16)

# Set image-related DICOM attributes
dataset.Rows = rows
dataset.Columns = columns
dataset.BitsAllocated = bits_allocated
dataset.SamplesPerPixel = samples_per_pixel
dataset.NumberOfFrames = 100
dataset.PixelData = pixel_array.tobytes()

# Save the DICOM dataset to a file
filename = "new_dicom_file_with_pixel_data.dcm"
pydicom.filewriter.write_file(filename, dataset)