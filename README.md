# MedMNIST 
## [arXiv Preprint](https://arxiv.org/abs/2010.14925) | [Project Page](https://medmnist.github.io/) | [Dataset](https://www.dropbox.com/sh/upxrsyb5v8jxbso/AADOV0_6pC9Tb3cIACro1uUPa?dl=0)
[Jiancheng Yang](https://jiancheng-yang.com/), Rui Shi, [Bingbing Ni](https://scholar.google.com/citations?user=eUbmKwYAAAAJ), [Bilian Ke](https://scholar.google.com/citations?user=2cX5y8kAAAAJ)

We present *MedMNIST*, a collection of 10 pre-processed medical open datasets. MedMNIST is standardized to perform classification tasks on lightweight 28 * 28 images, which requires no background knowledge. Covering the primary data modalities in medical image analysis, it is diverse on data scale (from 100 to 100,000) and tasks (binary/multi-class, ordinal regression and multi-label). MedMNIST could be used for educational purpose, rapid prototyping, multi-modal machine learning or AutoML in medical image analysis. Moreover, MedMNIST Classification Decathlon is designed to benchmark AutoML algorithms on all 10 datasets. 

![MedMNIST_Decathlon](overview.jpg)

More details, please refer to our paper:

**MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis** ([arXiv](https://arxiv.org/abs/2010.14925))

# Key Features
* ***Educational***: Our multi-modal data, from multiple open medical image datasets with Creative Commons (CC) Licenses, is easy to use for educational purpose.
* ***Standardized***: Data is pre-processed into same format, which requires no background knowledge for users.
* ***Diverse***: The multi-modal datasets covers diverse data scales (from 100 to 100,000) and tasks (binary/multiclass, ordinal regression and multi-label).
* ***Lightweight***: The small size of 28 × 28 is friendly for rapid prototyping and experimenting multi-modal machine learning and AutoML algorithms.

Please note that this dataset is **NOT** intended for clinical use.

# Code Structure
* [`medmnist/`](medmnist/):
    * [`dataset.py`](medmnist/dataset.py): dataloaders of medmnist.
    * [`models.py`](medmnist/models.py): *ResNet-18* and *ResNet-50* models.
    * [`evaluator.py`](medmnist/evaluator.py): evaluate metrics.
    * [`environ.py`](medmnist/environ.py): roots.
* [`train.py`](train.py): the training script.

# Requirements
* Python 3 (Anaconda 3.6.3 specifically)
* PyTorch\==0.3.1
* numpy\==1.18.5, pandas\==0.25.3, scikit-learn\==0.22.2

Higher versions should also work (perhaps with minor modifications).


# Dataset

Our MedMNIST dataset is available on [Dropbox](https://www.dropbox.com/sh/upxrsyb5v8jxbso/AADOV0_6pC9Tb3cIACro1uUPa?dl=0).

The dataset contains ten subsets, and each subset (e.g., `pathmnist.npz`) is comprised of `train_images`, `train_labels`, `val_images`, `val_labels`, `test_images` and `test_labels`.

# How to run the experiments

* Download Dataset  [MedMNIST](https://www.dropbox.com/sh/upxrsyb5v8jxbso/AADOV0_6pC9Tb3cIACro1uUPa?dl=0).

* Modify the paths

  Specify `dataroot` and `outputroot` in  [./medmnist/environ.py](./medmnist/environ.py) 

  `dataroot` is the root where you save our `npz` datasets

  `outputroot` is the root where you want to save testing results

* Run our [`train.py`](./train.py) script in terminal. 

  First, change directory to where train.py locates. Then, use command `python train.py xxxmnist` to run the experiments, where `xxxmnist` is subset of our MedMNIST (e.g., `pathmnist`).

# Citation
If you find this project useful, please cite our paper as:

      Jiancheng Yang, Rui Shi, Bingbing Ni. "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis," arXiv preprint arXiv:2010.14925, 2020.

or using bibtex:
     
     @article{medmnist,
     title={MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis},
     author={Yang, Jiancheng and Shi, Rui and Ni, Bingbing},
     journal={arXiv preprint arXiv:2010.14925},
     year={2020}
     }

# LICENSE
The code is under Apache-2.0 License.
