<div align="center">

# Re-implementation of ControlNet with Shape Masks

[[`Hugging Face`]]()
</div>

<!-- omit in toc -->
# Table of Contents
- [<u>1. Overview</u>](#overview)
- [<u>2. To-Do List</u>](#to-do-list)
- [<u>3. Code Structure</u>](#code-structure)
- [<u>4. Implementation Details</u>](#implementation-details)
- [<u>5. Prerequisites</u>](#prerequisites)
- [<u>6. Training</u>](#training)
- [<u>7. Sampling</u>](#sampling)
- [<u>8. Results</u>](#results)
- [<u>9. Citation</u>](#citation)
- [<u>10. Stars, Forked, and Star History</u>](#stars-forked-and-star-history)

<!-- omit in toc -->
# Overview
This is a re-implementation of ControlNet trained with shape masks.
If you have any questions about this work, please feel free to [start a new issue](https://github.com/AlonzoLeeeooo/shape-guided-controlnet/issues/new) or [propose a PR](https://github.com/AlonzoLeeeooo/shape-guided-controlnet/pulls).



<!-- omit in toc -->
# To-Do List
- Regular Maintainence

<!-- omit in toc -->
# Code Structure
```
├── LICENSE
├── README.md
├── annotators                       <----- Code of annotators for shape masks
│   └── u2net_saliency_detection
├── dataset_loaders                  <----- Code of dataset loaders
├── examples                         <----- Example conditions for validation use
│   └── conditions
├── inference.py                     <----- Script to inference trained ControlNet model
├── runners                          <----- Source code of training and inference runners
│   ├── controlnet_inference_runner.py
│   └── controlnet_train_runner.py
├── train.py                         <----- Script to train ControlNet model
└── utils                            <----- Code of toolkit functions
```


<!-- omit in toc -->
# Implementation Details
The ControlNet model is trained on COCO dataset with 100,000 iterations, along with a batch size of 4.
Each data sample consists of an image, a descriptive caption, and a shape mask.
The image caption directly uses the official annotations (i.e., `captions_train2014.json`) in COCO dataset.
For the shape mask, we use the off-the-shelf saliency detection model [`u2net`](https://github.com/xuebinqin/U-2-Net) to generate the shape mask for each image, where you can find more details in `annotators/u2net_saliency_detection`.

<!-- omit in toc -->
# Prerequisites
You can use the one-click installation script `install.sh` to install all the dependencies.

<!-- omit in toc -->
# Training


<!-- omit in toc -->
# Sampling

<!-- omit in toc -->
# Results

<!-- omit in toc -->
# Stars, Forked, and Star History