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

```python
writeToNpy('./data/training', './npdata/training', training=False)
writeToNpy('./data/testing', './npdata/testing', training=True)
writeToNpy('./data/validation', './npdata/validation', training=True)
```

## Citation
Please consider citing the project in your publications if it helps your research. The following is a BibTeX reference. The BibTeX entry requires the url LaTeX package.
```bash
writeToNpy('./data/training', './npdata/training', training=False)
writeToNpy('./data/testing', './npdata/testing', training=True)
writeToNpy('./data/validation', './npdata/validation', training=True)
```
