# How we generated the MedMNIST+ (the larger-size version of MedMNIST)

The data in MedMNIST+ directly corresponds to that of MedMNIST, maintaining the same dataset splits (i.e., train, val, and test) and sample indices. The primary distinction between the two lies in the image sizes. For 2D images, MedMNIST+ offers sizes of 28x28, 64x64, 128x128, and 224x224 pixels, whereas for 3D images, sizes of 28x28x28 and 64x64x64 are available.

It's crucial to note that in MedMNIST, where data is processed through center-cropping followed by resizing, MedMNIST+ employs the same initial center-crop size and then resizes the images to achieve the desired target resolutions.

We will describe the details of MedMNIST+, compared with MedMNIST. More details of the standard MedMNIST can be found in our [paper](https://doi.org/10.1038/s41597-022-01721-8).

### PathMNIST
In MedMNIST, we resize the source images of 3x224x224 into 3x28x28. 

In MedMNIST+, for size 224x224, we utilize the source images directly; while for size 64x64 and 128x128, we resize the source images of 3x224x224 into the target size. 

### ChestMNIST
In MedMNIST, we resize the source images of 1x1,024x1,024 into 1x28x28. 

In MedMNIST+, for size 224x224, 128x128, and 64x64, we resize the source images of 1x1,024x1,024 into the target size. 

### DermaMNIST
In MedMNIST, the source images of 3x600x450 are resized into 3x28x28. 

In MedMNIST+, for size 224x224, 128x128, and 64x64, we resize the source images of 3x600x450 into the target size. 

### OCTMNIST
In MedMNIST, the source images are gray-scale, and their sizes are (384–1,536)x(277–512). We center-crop the images with a window size of length of the short edge and resize them into 1x28x28. 

In MedMNIST+, for size 224x224, 128x128, and 64x64, we center-crop the images of (384–1,536)x(277–512) with a window size of length of the short edge, and then resize the source images into the target size. 

### PneumoniaMNIST
In MedMNIST, the source images are gray-scale, and their sizes are (384–2,916)x(127–2,713). We center-crop the images with a window size of length of the short edge and resize them into 1x28x28. 

In MedMNIST+, for size 224x224, 128x128, and 64x64, we center-crop the source images of (384–2,916)x(127–2,713) with a window size of length of the short edge, and then resize the source images into the target size. Note that after center-crop, the smallest size of the images is 127x127, and we upsample them to 128x128 and 224x224. 

### RetinaMNIST
In MedMNIST, the source images of 3x1,736x1,824 are center-cropped with a window size of length of the short edge and resized into 3x28x28. 

In MedMNIST+, for size 224x224, 128x128, and 64x64, we first center-crop the source images of 3x1,736x1,824 with a window size of length of the short edge, and then resize the source images into the target size. 

### BreastMNIST
In MedMNIST, the source images of 1x500x500 are resized into 1x28x28. 

In MedMNIST+, for size 224x224, 128x128, and 64x64, we resize the source images of 1x500x500 into the target size. 

### BloodMNIST
In MedMNIST, the source images with resolution 3x360x363 pixels are center-cropped into 3x200x200, and then resized into 3x28x28. 

In MedMNIST+, for size 224x224, 128x128, and 64x64, we first center-crop the images with resolution 3x360x363 into 3x200x200, and then resize the source images into the target size. 

### TissueMNIST
In MedMNIST, each gray-scale image is 32x32x7 pixels, where 7 denotes 7 slices. We obtain 2D maximum projections by taking the maximum pixel value along the axial-axis of each pixel, and resize them into 28x28 gray-scale images. 

In MedMNIST+, following the same procedure in MedMNIST, we obtain 2D maximum projections by taking the maximum pixel value along the axial-axis of each pixel, resulting in images of size 1x32x32. Then, for size 224x224, 128x128, and 64x64, we resize the source images into the target size. 

### Organ{A,C,S}MNIST
In MedMNIST, Hounsfeld-Unit (HU) of the 3D images are transformed into gray-scale with an abdominal window. We crop 2D images from the center slices of the 3D bounding boxes in axial/coronal/sagittal views (planes). The only diferences of Organ{A,C,S}MNIST are the views. The images are resized into 1x28x28 to perform multi-class classifcation of 11 body organs. 

In MedMNIST+, following the same procedure in MedMNIST, we crop 2D images from the center slices of the 3D bounding boxes in axial/coronal/sagittal views (planes). Then, for size 224x224, 128x128, and 64x64, we resize the source images into the target size. 

### OrganMNIST3D
In MedMNIST, the source of the OrganMNIST3D is the same as that of the Organ{A,C,S}MNIST. Instead of 2D images, we directly use the 3D bounding boxes of size (15-152)x(15-112)x(12-1145) and process the images into 28x28x28 to perform multi-class classifcation of 11 body organs. 

In MedMNIST+, we directly use the 3D bounding boxes of size (15-152)x(15-112)x(12-1145) and process the images into 64x64x64. 

### NoduleMNIST3D
In MedMNIST, we center-crop the spatially normalized images of size 80x80x80 into 28x28x28. 

In MedMNIST+, to be consistent with the dataset in MedMNIST, we center-crop the spatially normalized images of size 80x80x80 into 28x28x28, and then upsample them into 64x64x64.

### AdrenalMNIST3D
In MedMNIST, we center-crop the images of size 120x120x120 into 64x64x64, and then resize them into 28x28x28. 

In MedMNIST+, we center-crop the images of size 120x120x120 into 64x64x64.

### FractureMNIST3D
In MedMNIST, for each annotated fracture area, we calculate its center and resize the center-cropped 64x64x64 image into 28x28x28. 

In MedMNIST+, for each annotated fracture area, we calculate its center and center-crop the image into 64x64x64. 

### VesselMNIST3D
In MedMNIST, we fix the non-watertight mesh with PyMeshFix and voxelize the watertight mesh with trimesh into 28x28x28 voxels. 

In MedMNIST+, we fix the non-watertight mesh with PyMeshFix and voxelize the watertight mesh with trimesh into 64x64x64 voxels. 

### SynapseMNIST3D
In MedMNIST, the original data is of the size 100umx100umx100um and the resolution 8nmx8nmx30nm, where a 30umx30umx30um sub-volume was used in the MitoEM dataset with dense 3D mitochondria instance segmentation labels. For each labeled synaptic location, we crop a 3D volume of 1024nmx1024nmx1024nm and resize it into 28x28x28 voxels. 

In MedMNIST+, for each labeled synaptic location, we crop a 3D volume of 1024x1024x1024 and center-crop it into 64x64x64 voxels. 
