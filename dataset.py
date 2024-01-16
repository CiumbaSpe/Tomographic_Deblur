import os
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import mean_squared_error
import torchvision.transforms as TF

class SeeTrough2d(Dataset):
    def __init__(self, img_dir_x, img_dir_y, transform=None, target_transform=None):
        self.img_x = img_dir_x 
        self.img_y = img_dir_y
        self.transform = transform
        self.target_transform = target_transform
        self.images = sorted(os.listdir(img_dir_y))
 
    def __len__(self):
        #print("THE LEN OF THE DATASET: ", len(self.images), "\n")
        return len(self.images)
    
    def __getitem__(self, idx):
        sample_x = os.path.join(self.img_x, self.images[idx])
        target_y = os.path.join(self.img_y, self.images[idx])

        image = np.load(sample_x).astype(np.float32)
        target = np.load(target_y).astype(np.float32)

        # normalize 0-1 
        x = (image - np.min(image)) / (np.max(image) - np.min(image))
        y = (target - np.min(target)) / (np.max(target) - np.min(target))
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        
        return x, y

class SeeTrough3d(Dataset):
    def __init__(self, img_dir_x, img_dir_y, transform=None, target_transform=None):
        # self.orderd_x = sorted(os.listdir(img_dir_x))
        # self.orderd_y = sorted(os.listdir(img_dir_y))
        self.img_x = img_dir_x 
        self.img_y = img_dir_y
        self.transform = transform
        self.target_transform = target_transform
        self.images = sorted(os.listdir(img_dir_y))
 
    def __len__(self):
        #print("THE LEN OF THE DATASET: ", len(self.images), "\n")
        return len(self.images)
    
    def __getitem__(self, idx):

        concat_img = []
        concat_trg = []

        for i in range(4):

            # add this comment line for stride 4 in get_volume
            if(idx == 0 or (idx != 1 and idx % 4 == 0)):
                if(idx + i < self.__len__()):
                    sample_x = os.path.join(self.img_x, self.images[idx + i])
                    target_y = os.path.join(self.img_y, self.images[idx + i])

                    image = np.load(sample_x).astype(np.float32)
                    target = np.load(target_y).astype(np.float32)

                    concat_img.append(image)
                    concat_trg.append(target)
            else:
                return 0, 0, False

        x = np.stack(concat_img)
        y = np.stack(concat_trg)

        # normalize 0-1 
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        y = (y - np.min(y)) / (np.max(y) - np.min(y))
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        
        return x, y, idx


def main():
    a = SeeTrough3d("gigadose_dataset/trainIn", "gigadose_dataset/trainOut")
    for i in range(a.__len__()):
        x, y = a.__getitem__(i)
        if(x.shape[0] != 4):
            print("Broken ", i)

if __name__ == "__main__":
    main()
