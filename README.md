# MedMNIST: [medmnist.com](https://medmnist.github.io/)
## Data ([Zenodo](https://doi.org/10.5281/zenodo.6496656)) | Publication ([Scientific Data](https://doi.org/10.1038/s41597-022-01721-8) / [arXiv](https://arxiv.org/abs/2110.14795)) | MedMNIST v1 ([ISBI'21](https://medmnist.github.io/v1)) 
[Jiancheng Yang](https://jiancheng-yang.com/), Rui Shi, [Donglai Wei](https://donglaiw.github.io/), Zequan Liu, Lin Zhao, [Bilian Ke](https://scholar.google.com/citations?user=2cX5y8kAAAAJ&hl=en), [Hanspeter Pfister](https://scholar.google.com/citations?user=VWX-GMAAAAAJ&hl=en), [Bingbing Ni](https://scholar.google.com/citations?user=eUbmKwYAAAAJ)

[![](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MedMNIST/MedMNIST/blob/main/examples/getting_started.ipynb)

We introduce *MedMNIST v2*, a large-scale MNIST-like collection of standardized biomedical images, including 12 datasets for 2D and 6 datasets for 3D. All images are pre-processed into 28x28 (2D) or 28x28x28 (3D) with the corresponding classification labels, so that no background knowledge is required for users. Covering primary data modalities in biomedical images, MedMNIST v2 is designed to perform classification on lightweight 2D and 3D images with various data scales (from 100 to 100,000) and diverse tasks (binary/multi-class, ordinal regression and multi-label). The resulting dataset, consisting of 708,069 2D images and 9,998 3D images in total, could support numerous research / educational purposes in biomedical image analysis, computer vision and machine learning. We benchmark several baseline methods on MedMNIST v2, including 2D / 3D neural networks and open-source / commercial AutoML tools. 

![MedMNISTv2_overview](https://raw.githubusercontent.com/MedMNIST/MedMNIST/main/assets/medmnistv2.jpg)

For more details, please refer to our paper:

**MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification** ([Scientific Data](https://doi.org/10.1038/s41597-022-01721-8) / [arXiv](https://arxiv.org/abs/2110.14795))

or its conference version:

**MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis** ([ISBI'21](https://arxiv.org/abs/2010.14925))

# Key Features
* ***Diverse***: It covers diverse data modalities, dataset scales (from 100 to 100,000), and tasks (binary/multi-class, multi-label, and ordinal regression). It is as diverse as the VDD and MSD to fairly evaluate the generalizable performance of machine learning algorithms in different settings, but both 2D and 3D biomedical images are provided. 
* ***Standardized***: Each sub-dataset is pre-processed into the same format, which requires no background knowledge for users. As an MNIST-like dataset collection to perform classification tasks on small images, it primarily focuses on the machine learning part rather than the end-to-end system. Furthermore, we provide standard train-validation-test splits for all datasets in MedMNIST v2, therefore algorithms could be easily compared. 
* ***Lightweight***: The small size of 28×28 (2D) or 28×28×28 (3D) is friendly to evaluate machine learning algorithms. 
* ***Educational***: As an interdisciplinary research area, biomedical image analysis is difficult to hand on for researchers from other communities, as it requires background knowledge from computer vision, machine learning, biomedical imaging, and clinical science. Our data with the Creative Commons (CC) License is easy to use for educational purposes.

Please note that this dataset is **NOT** intended for clinical use.

# Code Structure
* [`medmnist/`](medmnist/):
    * [`dataset.py`](medmnist/dataset.py): PyTorch datasets and dataloaders of MedMNIST.
    * [`evaluator.py`](medmnist/evaluator.py): Standardized evaluation functions.
    * [`info.py`](medmnist/info.py): Dataset information `dict` for each subset of MedMNIST.
* [`examples/`](examples/):
    * [`getting_started.ipynb`](examples/getting_started.ipynb): To explore the MedMNIST dataset with jupyter notebook. It is ONLY intended for a quick exploration, i.e., it does not provide full training and evaluation functionalities. 
    * [`getting_started_without_PyTorch.ipynb`](examples/getting_started_without_PyTorch.ipynb): This notebook provides snippets about how to use MedMNIST data (the `.npz` files) without PyTorch.
* [`setup.py`](setup.py): To install `medmnist` as a module.
* [EXTERNAL] [`MedMNIST/experiments`](https://github.com/MedMNIST/experiments): training and evaluation scripts to reproduce both 2D and 3D experiments in our paper, including PyTorch, auto-sklearn, AutoKeras and Google AutoML Vision together with their weights ;)

# Installation and Requirements
Setup the required environments and install `medmnist` as a standard Python package from [PyPI](https://pypi.org/project/medmnist/):

    pip install medmnist

Or install from source:

    pip install --upgrade git+https://github.com/MedMNIST/MedMNIST.git

Check whether you have installed the latest [version](medmnist/info.py):

    >>> import medmnist
    >>> print(medmnist.__version__)

The code requires only common Python environments for machine learning. Basically, it was tested with
* Python 3 (>=3.6)
* PyTorch\==1.3.1
* numpy\==1.18.5, pandas\==0.25.3, scikit-learn\==0.22.2, Pillow\==8.0.1
* fire, scikit-image

Higher (or lower) versions should also work (perhaps with minor modifications). 

# If you use PyTorch...

* Great! Our code is designed to work with PyTorch.

* Explore the MedMNIST dataset with jupyter notebook ([`getting_started.ipynb`](examples/getting_started.ipynb)), and train basic neural networks in PyTorch.


# If you do not use PyTorch...

* Although our code is tested with PyTorch, you are free to parse them with your own code (without PyTorch or even without Python!), as they are only standard NumPy serialization files. It is simple to create a dataset without PyTorch.
* Go to [`getting_started_without_PyTorch.ipynb`](examples/getting_started_without_PyTorch.ipynb), which provides snippets about how to use MedMNIST data (the `.npz` files) without PyTorch.
* Simply change the super class of `MedMNIST` from `torch.utils.data.Dataset` to `collections.Sequence`, you will get a standard dataset without PyTorch. Check [`dataset_without_pytorch.py`](examples/dataset_without_pytorch.py) for more details.
* You still have most functionality of our MedMNIST code ;)

# Dataset

Please download the dataset(s) via [`Zenodo`](https://doi.org/10.5281/zenodo.6496656). You could also use our code to download automatically by setting `download=True` in [`dataset.py`](medmnist/dataset.py).

The MedMNIST dataset contains several subsets. Each subset (e.g., `pathmnist.npz`) is comprised of 6 keys: `train_images`, `train_labels`, `val_images`, `val_labels`, `test_images` and `test_labels`.
* `train_images` / `val_images` / `test_images`: `N` × 28 × 28 for 2D gray-scale datasets, `N` × 28 × 28 × 3 for 2D RGB datasets, `N` × 28 × 28 × 28 for 3D datasets. `N` denotes the number of samples.  
* `train_labels` / `val_labels` / `test_labels`: `N` x `L`. `N` denotes the number of samples. `L` denotes the number of task labels; for single-label (binary/multi-class) classification, `L=1`, and `{0,1,2,3,..,C}` denotes the category labels (`C=1` for binary); for multi-label classification `L!=1`, e.g., `L=14` for `chestmnist.npz`.

# Command Line Tools

* List all available datasets:
    
        python -m medmnist available

* Download all available datasets:
    
        python -m medmnist download

* Delete all downloaded npz from root:

        python -m medmnist clean

* Print the dataset details given a subset flag:

        python -m medmnist info --flag=xxxmnist

* Save the dataset as standard figure and csv files, which could be used for AutoML tools, e.g., Google AutoML Vision:

    for 2D datasets:

        python -m medmnist save --flag=xxxmnist --folder=tmp/ --postfix=png
        
    for 3D datasets:

        python -m medmnist save --flag=xxxmnist3d --folder=tmp/ --postfix=gif

* Parse and evaluate a standard result file, refer to [`Evaluator.parse_and_evaluate`](medmnist/evaluator.py) for details.

        python -m medmnist evaluate --path=folder/{flag}_{split}@{run}.csv

# License and Citation

The code is under [Apache-2.0 License](./LICENSE).

The MedMNIST dataset is licensed under *Creative Commons Attribution 4.0 International* ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)), except DermaMNIST under 
*Creative Commons Attribution-NonCommercial 4.0 International* ([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)).

If you find this project useful in your research, please cite the following papers:

    Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni. Yang, Jiancheng, et al. "MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification." Scientific Data, 2023.

    Jiancheng Yang, Rui Shi, Bingbing Ni. "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis". IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021.

or using the bibtex:

    @article{medmnistv2,
        title={MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification},
        author={Yang, Jiancheng and Shi, Rui and Wei, Donglai and Liu, Zequan and Zhao, Lin and Ke, Bilian and Pfister, Hanspeter and Ni, Bingbing},
        journal={Scientific Data},
        volume={10},
        number={1},
        pages={41},
        year={2023},
        publisher={Nature Publishing Group UK London}
    }
     
    @inproceedings{medmnistv1,
        title={MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis},
        author={Yang, Jiancheng and Shi, Rui and Ni, Bingbing},
        booktitle={IEEE 18th International Symposium on Biomedical Imaging (ISBI)},
        pages={191--195},
        year={2021}
    }

Please also cite source data paper(s) of the MedMNIST subset(s) as per the description on the [project page](https://medmnist.github.io/).

# Release Notes
* `v2.2.3`: DermaMNIST license to CC BY-NC 4.0
* `v2.2.2`: Python 3.11 `Sequence` from collections.abc supported
* `v2.2.1`: PyPI info updated
* `v2.2.0`: `montage` method supported for scikit-image>=0.20.0
* `v2.1.0`: `NoduleMNIST3D` data error fixed
* `v2.0.0`: MedMNIST v2 release