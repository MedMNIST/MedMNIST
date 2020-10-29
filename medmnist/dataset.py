from medmnist import environ
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


INFO = "medmnist/medmnist.json"


class MedMNIST(Dataset):

    flag = ...

    def __init__(self, split='train', transform=None, target_transform=None):
        ''' dataset
        :param split: 'train', 'val' or 'test', select dataset
        :param transform: data transformation
        :param target_transform: target transformation
    
        '''

        npz_file = np.load(os.path.join(environ.dataroot,"{}.npz".format(self.flag)))

        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        if self.split == 'train':
            self.img = npz_file['train_images']
            self.label = npz_file['train_labels']
        elif self.split == 'val':
            self.img = npz_file['val_images']
            self.label = npz_file['val_labels']
        elif self.split == 'test':
            self.img = npz_file['test_images']
            self.label = npz_file['test_labels']

    def __getitem__(self, index):
        img, target = self.img[index], int(self.label[index])
        img = Image.fromarray(np.uint8(img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return self.img.shape[0]


class PathMNIST(MedMNIST):
    flag = "pathmnist"


class OCTMNIST(MedMNIST):
    flag = "octmnist"


class PneumoniaMNIST(MedMNIST):
    flag = "pneumoniamnist"


class ChestMNIST(MedMNIST):
    flag = "chestmnist"


class DermaMNIST(MedMNIST):
    flag = "dermamnist"


class RetinaMNIST(MedMNIST):
    flag = "retinamnist"


class BreastMNIST(MedMNIST):
    flag = "breastmnist"


class OrganMNIST_Axial(MedMNIST):
    flag = "organmnist_axial"


class OrganMNIST_Coronal(MedMNIST):
    flag = "organmnist_coronal"


class OrganMNIST_Sagittal(MedMNIST):
    flag = "organmnist_sagittal"
