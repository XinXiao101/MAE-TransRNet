# MAE-TransRNet
Transformer-ConvNet with Masked AutoEncoder for Cardiac Image Registration
## Overview
This repository provides the method described in the paper
> XX, et al. "MAE-TransRNet: An improved Transformer-ConvNet architecture with Masked AutoEncoder for Cardiac MRI Registration"
## Requirements
Python 3.8, PyTorch 1.8.2 and other common packages are listed in [requirements.txt](https://github.com/SuperNatural-101/MAE-TransRNet/blob/main/requirements.txt)
## Description
The repository gives an improved Transformer-ConvNet architecture with Masked AutoEncoder for Cardiac MRI Registration. The core of the Transformer is designed as a Masked AutoEncoder (MAE) and a lightweight decoder structure, and feature extraction before the downstream registration task is transformed into the self-supervised learning task. The proposed network employs the latent representation and masked token learned from autoencoder by masking a high proportion of the image patches to restore the semantic information of the original image. Subsequently, the advantages of the extraction of local features from CNN are combined with a MAE, and the proposed network is applied to the downstream cardiac image registration task. This study also attempts to embed different attention mechanisms modules into CNN and Transformer structures to better represent their feature maps, so as to highlight image details and maintain high spatial resolution image features. 
## Usage
### Preprocessing of ACDC Data
#### MAE architecture for Pre-training
For the pre-training task, we pre-processed the ACDC dataset in this way: we divided the 150 (100+50) cases of nii data into three folders of ImagesTr, ImagesTs, and labelsTr according to a JSON file which is provided which contains the training and validation splits that were used for the training. The json file can be found in the json_files directory of the json_files.
#### Downstream task
the dataset applied is the publicly available benchmark dataset from the Automated Cardiac Diagnosis Challenge(ACDC) in 2017. This dataset contains short-axis cardiac 3D MR images from a total of 150 cases for two time frames of initial frame-end frame, and the public dataset applied provides standard segmentation labels for three parts (including the left ventricle (LV), the left ventricular myocardium (Myo), and the right ventricle (RV)) for the registration task, which involves five categories of cases (including normal, heart failure with infarction, dilated cardiomyopathy, hypertrophic cardiomyopathy, and right ventricular abnormalities). 100 cases of the above 150 cases contain the triple segmentation labels, while 50 cases do not contain labels. 

At the data preprocessing stage, all the images are cropped to $64\times 128\times 128$, the random flip is adopted as the data augment method for the training set to increase the sample size of the dataset. Furthermore, the label pixel normalization method is applied for the validation and test sets to preprocess the data.

Both training data, validation data and test data are transformed as .npy files through preprocess.py.

```python
writeToNpy('./data/training', './npdata/training', training=False)
writeToNpy('./data/testing', './npdata/testing', training=True)
writeToNpy('./data/validation', './npdata/validation', training=True)
```
### Training
#### MAE Pretraining

#### Registration task
