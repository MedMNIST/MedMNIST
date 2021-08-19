import os
from sys import base_prefix
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from medmnist.info import INFO, HOMEPAGE, DEFAULT_ROOT


class MedMNIST(Dataset):

    flag = ...

    def __init__(self,
                 split,
                 transform=None,
                 target_transform=None,
                 download=False,
                 as_rgb=False,
                 root=DEFAULT_ROOT):
        ''' dataset
        :param split: 'train', 'val' or 'test', select subset
        :param transform: data transformation
        :param target_transform: target transformation

        '''

        self.info = INFO[self.flag]

        if root is not None and os.path.exists(root):
            self.root = root
        else:
            raise RuntimeError("Failed to setup the default `root` directory. " +
                               "Please specify and create the `root` directory manually.")

        if download:
            self.download()

        if not os.path.exists(
                os.path.join(self.root, "{}.npz".format(self.flag))):
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        npz_file = np.load(os.path.join(self.root, "{}.npz".format(self.flag)))

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

        self.as_rgb = as_rgb

    def __len__(self):
        return self.img.shape[0]

    def __repr__(self):
        '''Adapted from torchvision.ss'''
        _repr_indent = 4
        head = f"Dataset {self.__class__.__name__} ({self.flag})"
        body = [f"Number of datapoints: {self.__len__()}"]
        body.append(f"Root location: {self.root}")
        body.append(f"Split: {self.split}")
        body.append(f"Task: {self.info['task']}")
        body.append(f"Number of channels: {self.info['n_channels']}")
        body.append(f"Meaning of labels: {self.info['label']}")
        body.append(f"Number of samples: {self.info['n_samples']}")
        body.append(f"Description: {self.info['description']}")
        body.append(f"License: {self.info['license']}")

        lines = [head] + [" " * _repr_indent + line for line in body]
        return '\n'.join(lines)

    def download(self):
        try:
            from torchvision.datasets.utils import download_url
            download_url(url=self.info["url"],
                         root=self.root,
                         filename="{}.npz".format(self.flag),
                         md5=self.info["MD5"])
        except:
            raise RuntimeError('Something went wrong when downloading! ' +
                               'Go to the homepage to download manually. ' +
                               HOMEPAGE)


class MedMNIST2D(MedMNIST):

    def __getitem__(self, index):
        img, target = self.img[index], self.label[index].astype(int)
        img = Image.fromarray(img)

        if self.as_rgb:
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def save(self, folder, postfix="png", write_csv=True):

        split_dict = {
            "train": "TRAIN",
            "val": "VALIDATION",
            "test": "TEST"
        }  # compatible for Google AutoML Vision

        from tqdm import trange

        _transform = self.transform
        _target_transform = self.target_transform
        self.transform = None
        self.target_transform = None

        base_folder = os.path.join(folder, self.flag)

        if not os.path.exists(base_folder):
            os.makedirs(base_folder)

        if write_csv:
            csv_file = open(os.path.join(folder, f"{self.flag}.csv"), "a")

        for idx in trange(self.__len__()):

            img, label = self.__getitem__(idx)

            file_name = f"{self.split}{idx}_{'_'.join(map(str,label))}.{postfix}"

            img.save(os.path.join(base_folder, file_name))

            if write_csv:
                line = f"{split_dict[self.split]},{file_name},{','.join(map(str,label))}\n"
                csv_file.write(line)

        self.transform = _transform
        self.target_transform = _target_transform
        csv_file.close()

    def montage(self, length=20, replace=False, save_folder=None):
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from skimage.util import montage as skimage_montage

        n_imgs = length * length
        sel = np.random.choice(self.__len__(), size=n_imgs, replace=replace)
        sel_img = self.img[sel]
        if self.info['n_channels'] == 3:
            montage_arr = skimage_montage(sel_img, multichannel=True)
        else:
            assert self.info['n_channels'] == 1
            montage_arr = skimage_montage(sel_img, multichannel=False)

        montage_img = Image.fromarray(montage_arr)

        if save_folder is not None:
            montage_img.save(
                os.path.join(save_folder,
                             f"{self.flag}_{self.split}_montage.jpg"))

        return montage_img


class MedMNIST3D(MedMNIST):

    def __getitem__(self, index):
        return super().__getitem__(index)

    def save(self, folder, postfix="png", write_csv=True):
        raise NotImplementedError

    def montage(self, length=20, replace=False, save_folder=None):
        raise NotImplementedError


class PathMNIST(MedMNIST2D):
    flag = "pathmnist"


class OCTMNIST(MedMNIST2D):
    flag = "octmnist"


class PneumoniaMNIST(MedMNIST2D):
    flag = "pneumoniamnist"


class ChestMNIST(MedMNIST2D):
    flag = "chestmnist"


class DermaMNIST(MedMNIST2D):
    flag = "dermamnist"


class RetinaMNIST(MedMNIST2D):
    flag = "retinamnist"


class BreastMNIST(MedMNIST2D):
    flag = "breastmnist"


class BloodMNIST(MedMNIST2D):
    flag = "bloodmnist"


class TissueMNIST(MedMNIST2D):
    flag = "tissuemnist"


class OrganAMNIST(MedMNIST2D):
    flag = "organamnist"


class OrganCMNIST(MedMNIST2D):
    flag = "organcmnist"


class OrganSMNIST(MedMNIST2D):
    flag = "organsmnist"


class OrganMNIST3D(MedMNIST3D):
    flag = "organmnist3d"


class NoduleMNIST3D(MedMNIST3D):
    flag = "nodulemnist3d"


class AdrenalMNIST3D(MedMNIST3D):
    flag = "adrenalmnist3d"


class FractureMNIST3D(MedMNIST3D):
    flag = "fracturemnist3d"


class VesselMNIST3D(MedMNIST3D):
    flag = "vesselmnist3d"


class SynapseMNIST3D(MedMNIST3D):
    flag = "synapsemnist3d"


# backward-compatible
OrganMNISTAxial = OrganAMNIST
OrganMNISTCoronal = OrganCMNIST
OrganMNISTSagittal = OrganSMNIST
