# Density Uncertainty Layers
This repository contains the PyTorch implementation for the paper "Density Uncertainty Layers for Reliable Uncertainty Estimation" published in AISTATS 2024. 

## Requirements
The code is implemented using PyTorch 1.12.1. 

Install the required packages
```
pip install -r requirements.txt
```

## Running Experiments
To train Rank1 Density Uncertainty Layers WRN28 on CIFAR-10/100, 
```
python run_cifar.py --dataset={cifar10/cifar100} --model=rank1_density_wrn28
```

To train other models, simply replace the model argument with 
```
density_wrn28, bayesian_wrn28, mcdropout_wrn28, vdropout_wrn28, rank1_wrn28
```

## Citation
```
@inproceedings{park2024,
  title={Density Uncertainty Layers for Reliable Uncertainty Estimation},
  author={Park, Yookoon and Blei, David},
  booktitle={AISTATS},
  year={2024}
}
```
