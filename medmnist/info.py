__version__ = "1.3"


import os
from os.path import expanduser
import warnings


def get_default_root():
    home = expanduser("~")
    dirpath = os.path.join(home, ".medmnist")

    try:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
    except:
        warnings.warn("Failed to setup default root.")
        dirpath = None

    return dirpath


DEFAULT_ROOT = get_default_root()

HOMEPAGE = "https://github.com/MedMNIST/MedMNIST/"

INFO = {
    "pathmnist": {
        "python_class": "PathMNIST",
        "description":
        "A dataset based on a prior study for predicting survival from colorectal cancer histology slides, which provides a dataset NCT-CRC-HE-100K of 100,000 non-overlapping image patches from hematoxylin & eosin stained histological images, and a test dataset CRC-VAL-HE-7K of 7,180 image patches from a different clinical center. 9 types of tissues are involved, resulting a multi-class classification task. We resize the source images of 3 x 224 x 224 into 3 x 28 x 28, and split NCT-CRC-HE-100K into training and valiation set with a ratio of 9:1.",
        "url":
        "https://zenodo.org/record/5208230/files/pathmnist.npz?download=1",
        "MD5": "a8b06965200029087d5bd730944a56c1",
        "task": "multi-class",
        "label": {
            "0": "adipose",
            "1": "background",
            "2": "debris",
            "3": "lymphocytes",
            "4": "mucus",
            "5": "smooth muscle",
            "6": "normal colon mucosa",
            "7": "cancer-associated stroma",
            "8": "colorectal adenocarcinoma epithelium"
        },
        "n_channels": 3,
        "n_samples": {
            "train": 89996,
            "val": 10004,
            "test": 7180
        },
        "license": "CC BY 4.0"
    },
    "chestmnist": {
        "python_class": "ChestMNIST",
        "description":
        "A dataset based on NIH-ChestXray14 dataset, comprising 112,120 frontal-view X-ray images of 30,805 unique patients with the text-mined 14 disease image labels, which could be formulized as multi-label binary classification task. We use the official data split, and resize the source images of 1 x 1024 x 1024 into 1 x 28 x 28.",
        "url":
        "https://zenodo.org/record/5208230/files/chestmnist.npz?download=1",
        "MD5": "02c8a6516a18b556561a56cbdd36c4a8",
        "task": "multi-label, binary-class",
        "label": {
            "0": "atelectasis",
            "1": "cardiomegaly",
            "2": "effusion",
            "3": "infiltration",
            "4": "mass",
            "5": "nodule",
            "6": "pneumonia",
            "7": "pneumothorax",
            "8": "consolidation",
            "9": "edema",
            "10": "emphysema",
            "11": "fibrosis",
            "12": "pleural",
            "13": "hernia"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 78468,
            "val": 11219,
            "test": 22433
        },
        "license": "CC0 1.0"
    },
    "dermamnist": {
        "python_class": "DermaMNIST",
        "description":
        "A dataset based on HAM10000, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. The dataset consists of 10,015 dermatoscopic images labeled as 7 different categories, as a multi-class classification task. We split the images into training, validation and test set with a ratio of 7:1:2. The source images of 3 x 600 x 450 are resized into 3 x 28 x 28.",
        "url":
        "https://zenodo.org/record/5208230/files/dermamnist.npz?download=1",
        "MD5": "0744692d530f8e62ec473284d019b0c7",
        "task": "multi-class",
        "label": {
            "0": "actinic keratoses and intraepithelial carcinoma",
            "1": "basal cell carcinoma",
            "2": "benign keratosis-like lesions",
            "3": "dermatofibroma",
            "4": "melanoma",
            "5": "melanocytic nevi",
            "6": "vascular lesions"
        },
        "n_channels": 3,
        "n_samples": {
            "train": 7007,
            "val": 1003,
            "test": 2005
        },
        "license": "CC BY-NC 4.0"
    },
    "octmnist": {
        "python_class": "OCTMNIST",
        "description":
        "A dataset based on a prior dataset of 109,309 valid optical coherence tomography (OCT) images for retinal diseases. 4 types are involved, leading to a multi-class classification task. We split the source training set with a ratio of 9:1 into training and validation set, and use its source validation set as the test set. The source images are single-channel, and their sizes range from (384-1,536) x (277-512). We center-crop the images and resize them into 1 x 28 x 28.",
        "url":
        "https://zenodo.org/record/5208230/files/octmnist.npz?download=1",
        "MD5": "c68d92d5b585d8d81f7112f81e2d0842",
        "task": "multi-class",
        "label": {
            "0": "choroidal neovascularization",
            "1": "diabetic macular edema",
            "2": "drusen",
            "3": "normal"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 97477,
            "val": 10832,
            "test": 1000
        },
        "license": "CC BY 4.0"
    },
    "pneumoniamnist": {
        "python_class": "PneumoniaMNIST",
        "description":
        "A dataset based on a prior dataset of 5,856 pediatric chest X-ray images. The task is binary-class classification of pneumonia and normal. We split the source training set with a ratio of 9:1 into training and validation set, and use its source validation set as the test set. The source images are single-channel, and their sizes range from (384-2,916) x (127-2,713). We center-crop the images and resize them into 1 x 28 x 28.",
        "url":
        "https://zenodo.org/record/5208230/files/pneumoniamnist.npz?download=1",
        "MD5": "28209eda62fecd6e6a2d98b1501bb15f",
        "task": "binary-class",
        "label": {
            "0": "normal",
            "1": "pneumonia"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 4708,
            "val": 524,
            "test": 624
        },
        "license": "CC BY 4.0"
    },
    "retinamnist": {
        "python_class": "RetinaMNIST",
        "description":
        "A dataset based on DeepDRiD, a dataset of 1,600 retina fundus images. The task is ordinal regression for 5-level grading of diabetic retinopathy severity. We split the source training set with a ratio of 9:1 into training and validation set, and use the source validation set as test set. The source images of 3 x 1,736 x 1,824 are center-cropped and resized into 3 x 28 x 28",
        "url":
        "https://zenodo.org/record/5208230/files/retinamnist.npz?download=1",
        "MD5": "bd4c0672f1bba3e3a89f0e4e876791e4",
        "task": "ordinal-regression",
        "label": {
            "0": "0",
            "1": "1",
            "2": "2",
            "3": "3",
            "4": "4"
        },
        "n_channels": 3,
        "n_samples": {
            "train": 1080,
            "val": 120,
            "test": 400
        },
        "license": "CC BY 4.0"
    },
    "breastmnist": {
        "python_class": "BreastMNIST",
        "description":
        "A dataset based on a dataset of 780 breast ultrasound images. It is categorized into 3 classes: normal, benign and malignant. As we use low-resolution images, we simplify the task into binary classification by combing normal and benign as negative, and classify them against malignant as positive. We split the source dataset with a ratio of 7:1:2 into training, validation and test set. The source images of 1 x 500 x 500 are resized into 1 x 28 x 28.",
        "url":
        "https://zenodo.org/record/5208230/files/breastmnist.npz?download=1",
        "MD5": "750601b1f35ba3300ea97c75c52ff8f6",
        "task": "binary-class",
        "label": {
            "0": "malignant",
            "1": "normal, benign"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 546,
            "val": 78,
            "test": 156
        },
        "license": "CC BY 4.0"
    },
    "bloodmnist": {
        "python_class": "BloodMNIST",
        "description":
        "A dataset based on a dataset of individual normal cells, captured from individuals without infection, hematologic or oncologic disease and free of any pharmacologic treatment at the moment of blood collection. It contains a total of 17,092 images and is organized into eight groups. We split the source dataset with a ratio of 7:1:2 into training, validation and test set. The source images with resolution 3 × 360 × 363 pixels are center-cropped into 3 × 200 × 200, and then resized into 3 × 28 × 28.",
        "url":
        "https://zenodo.org/record/5208230/files/bloodmnist.npz?download=1",
        "MD5": "7053d0359d879ad8a5505303e11de1dc",
        "task": "multi-class",
        "label": {
            "0": "basophil",
            "1": "eosinophil",
            "2": "erythroblast",
            "3": "ig",
            "4": "lymphocyte",
            "5": "monocyte",
            "6": "neutrophil",
            "7": "platelet"
        },
        "n_channels": 3,
        "n_samples": {
            "train": 11959,
            "val": 1712,
            "test": 3421
        },
        "license": "CC BY 4.0"
    },
    "tissuemnist": {
        "python_class": "TissueMNIST",
        "description":
        "A dataset based on image set BBBC051, available from the Broad Bioimage Benchmark Collection. The dataset contains 236,386 human kidney cortex cells, segmented from three reference tissue specimens and organized into eight categories. We split the source dataset with a ratio of 7:1:2 into training, validation and test set. Each grayscale image is 32 × 32 × 7 pixels. We take maximum in the third channel and resize them into 28 × 28 grayscale images.",
        "url":
        "https://zenodo.org/record/5208230/files/tissuemnist.npz?download=1",
        "MD5": "ebe78ee8b05294063de985d821c1c34b",
        "task": "multi-class",
        "label": {
            "0": "Collecting Duct, Connecting Tubule",
            "1": "Distal Convoluted Tubule",
            "2": "Glomerular endothelial cells",
            "3": "Interstitial endothelial cells",
            "4": "Leukocytes",
            "5": "Podocytes",
            "6": "Proximal Tubule Segments",
            "7": "Thick Ascending Limb"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 165466,
            "val": 23640,
            "test": 47280
        },
        "license": "CC BY 3.0"
    },
    "organamnist": {
        "python_class": "OrganAMNIST",
        "description":
        "A dataset based on 3D computed tomography (CT) images from Liver Tumor Segmentation Benchmark (LiTS). We use bounding-box annotations of 11 body organs from another study to obtain the organ labels. Hounsfield-Unit (HU) of the 3D images are transformed into grey scale with a abdominal window; we then crop 2D images from the center slices of the 3D bounding boxes in axial views (planes). The images are resized into 1 x 28 x 28 to perform multi-class classification of 11 body organs. 115 and 16 CT scans from the source training set are used as training and validation set, respectively. The 70 CT scans from the source test set are treated as the test set.",
        "url":
        "https://zenodo.org/record/5208230/files/organamnist.npz?download=1",
        "MD5": "866b832ed4eeba67bfb9edee1d5544e6",
        "task": "multi-class",
        "label": {
            "0": "bladder",
            "1": "femur-left",
            "2": "femur-right",
            "3": "heart",
            "4": "kidney-left",
            "5": "kidney-right",
            "6": "liver",
            "7": "lung-left",
            "8": "lung-right",
            "9": "pancreas",
            "10": "spleen"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 34581,
            "val": 6491,
            "test": 17778
        },
        "license": "CC BY 4.0"
    },
    "organcmnist": {
        "python_class": "OrganCMNIST",
        "description":
        "A dataset based on 3D computed tomography (CT) images from Liver Tumor Segmentation Benchmark (LiTS). We use bounding-box annotations of 11 body organs from another study to obtain the organ labels. Hounsfield-Unit (HU) of the 3D images are transformed into grey scale with a abdominal window; we then crop 2D images from the center slices of the 3D bounding boxes in coronal views (planes). The images are resized into 1 x 28 x 28 to perform multi-class classification of 11 body organs. 115 and 16 CT scans from the source training set are used as training and validation set, respectively. The 70 CT scans from the source test set are treated as the test set.",
        "url":
        "https://zenodo.org/record/5208230/files/organcmnist.npz?download=1",
        "MD5": "0afa5834fb105f7705a7d93372119a21",
        "task": "multi-class",
        "label": {
            "0": "bladder",
            "1": "femur-left",
            "2": "femur-right",
            "3": "heart",
            "4": "kidney-left",
            "5": "kidney-right",
            "6": "liver",
            "7": "lung-left",
            "8": "lung-right",
            "9": "pancreas",
            "10": "spleen"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 13000,
            "val": 2392,
            "test": 8268
        },
        "license": "CC BY 4.0"
    },
    "organsmnist": {
        "python_class": "OrganSMNIST",
        "description":
        "A dataset based on 3D computed tomography (CT) images from Liver Tumor Segmentation Benchmark (LiTS). We use bounding-box annotations of 11 body organs from another study to obtain the organ labels. Hounsfield-Unit (HU) of the 3D images are transformed into grey scale with a abdominal window; we then crop 2D images from the center slices of the 3D bounding boxes in sagittal views (planes). The images are resized into 1 x 28 x 28 to perform multi-class classification of 11 body organs. 115 and 16 CT scans from the source training set are used as training and validation set, respectively. The 70 CT scans from the source test set are treated as the test set.",
        "url":
        "https://zenodo.org/record/5208230/files/organsmnist.npz?download=1",
        "MD5": "e5c39f1af030238290b9557d9503af9d",
        "task": "multi-class",
        "label": {
            "0": "bladder",
            "1": "femur-left",
            "2": "femur-right",
            "3": "heart",
            "4": "kidney-left",
            "5": "kidney-right",
            "6": "liver",
            "7": "lung-left",
            "8": "lung-right",
            "9": "pancreas",
            "10": "spleen"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 13940,
            "val": 2452,
            "test": 8829
        },
        "license": "CC BY 4.0"
    },
    "organmnist3d": {
        "python_class": "OrganMNIST3D",
        "description":
        "A dataset based on 3D computed tomography (CT) images from Liver Tumor Segmentation Benchmark (LiTS). We use bounding-box annotations of 11 body organs from another study to obtain the organ labels. Hounsfield-Unit (HU) of the 3D images are transformed into grey scale with a abdominal window. The images are resized into 28 × 28 × 28 to perform multi-class classification of 11 body organs. 115 and 16 CT scans from the source training set are used as training and validation set, respectively. The 70 CT scans from the source test set are treated as the test set.",
        "url":
        "https://zenodo.org/record/5208230/files/organmnist3d.npz?download=1",
        "MD5": "21f0a239e7f502e6eca33c3fc453c0b6",
        "task": "multi-class",
        "label": {
            "0": "liver",
            "1": "kidney-right",
            "2": "kidney-left",
            "3": "femur-right",
            "4": "femur-left",
            "5": "bladder",
            "6": "heart",
            "7": "lung-right",
            "8": "lung-left",
            "9": "spleen",
            "10": "pancreas"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 972,
            "val": 161,
            "test": 610
        },
        "license": "CC BY 4.0"
    },
    "nodulemnist3d": {
        "python_class": "NoduleMNIST3D",
        "description":
        "A dataset based on LIDC-IDRI, a large public lung nodule dataset, containing images from clinical thoracic CT scan. The dataset is designed for both lung nodule segmentation and 5-level malignancy classification task. To perform binary classification, we categorize cases with maligancany level 1/2 into negative class and 4/5 into positive class, ignoring cases with malignancy level 3. We split the source dataset with a ratio of 7:1:2 into training, validation and test set, and center-crop the data into 28 × 28 × 28.",
        "url":
        "https://zenodo.org/record/5208230/files/nodulemnist3d.npz?download=1",
        "MD5": "902d495e3d91ad1a7bcac1a6b58a8fa2",
        "task": "binary-class",
        "label": {
            "0": "benign",
            "1": "malignant"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 1158,
            "val": 165,
            "test": 526
        },
        "license": "CC BY 3.0"
    },
    "adrenalmnist3d": {
        "python_class": "AdrenalMNIST3D",
        "description":
        "A dataset based on the shape of adrenal dataset. We calculate the center of adrenal and crop the area with radius 32 around the center. We resize the cropped 64 × 64 × 64 image into 28 × 28 × 28, and use the official split of training set, validation set and test set.",
        "url":
        "https://zenodo.org/record/5208230/files/adrenalmnist3d.npz?download=1",
        "MD5": "bbd3c5a5576322bc4cdfea780653b1ce",
        "task": "binary-class",
        "label": {
            "0": "normal",
            "1": "hyperplasia"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 1188,
            "val": 98,
            "test": 298
        },
        "license": "CC BY 3.0"
    },
    "fracturemnist3d": {
        "python_class": "FractureMNIST3D",
        "description":
        "A dataset based on RibFrac Dataset, containing around 5,000 rib fractures from 660 computed tomography (CT) scans. The dataset organize detected rib fractures into 4 clinical categories(buckle, nondisplaced, displaced or segmental rib fractures). As we use low-resolution images, we combine displaced and segmental rib fractures as class '3', and classify them against buckle(class 1) and nondisplaced(class 2). For each annotated fracture area, we calculate its center and crop the area with radius 32 around the center. We resize the cropped 64 × 64 × 64 image into 28 × 28 × 28, and use the official split of training set, validation set and test set.",
        "url":
        "https://zenodo.org/record/5208230/files/fracturemnist3d.npz?download=1",
        "MD5": "6aa7b0143a6b42da40027a9dda61302f",
        "task": "multi-class",
        "label": {
            "0": "buckle rib fracture",
            "1": "nondisplaced rib fracture",
            "2": "displaced rib fracture"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 1027,
            "val": 103,
            "test": 240
        },
        "license": "CC BY-NC 4.0"
    },
    "vesselmnist3d": {
        "python_class": "VesselMNIST3D",
        "description":
        "A dataset based on an open-access 3D intracranial aneurysm dataset, IntrA, containing 103 3D models of entire brain vessels collected by reconstructing scanned 2D MRA images of patients. 1,694 healthy vessel segments and 215 aneurysm segments are generated automatically from the complete models. We fix the non-watertight mesh with PyMeshFix and voxelize the watertight mesh with trimesh into 28 × 28 × 28 voxels. We split the source dataset with a ratio of 7:1:2 into training, validation and test set.",
        "url":
        "https://zenodo.org/record/5208230/files/vesselmnist3d.npz?download=1",
        "MD5": "2ba5b80617d705141f3f85627108fce8",
        "task": "binary-class",
        "label": {
            "0": "vessel",
            "1": "aneurysm"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 1335,
            "val": 192,
            "test": 382
        },
        "license": "CC0 1.0"
    },
    "synapsemnist3d": {
        "python_class": "SynapseMNIST3D",
        "description":
        "A dataset containing 1,285 excitatory synapse images and 474 inhibitory synapse images, with shape 34 × 128 × 128. The images are resized into 28 × 28 × 28 and split with a ratio of 7:1:2 into training, validation and test set.",
        "url":
        "https://zenodo.org/record/5208230/files/synapsemnist3d.npz?download=1",
        "MD5": "1235b78a3cd6280881dd7850a78eadb6",
        "task": "binary-class",
        "label": {
            "0": "inhibitory synapse",
            "1": "excitatory synapse"
        },
        "n_channels": 1,
        "n_samples": {
            "train": 1230,
            "val": 177,
            "test": 352
        },
        "license": "CC BY 4.0"
    }
}
