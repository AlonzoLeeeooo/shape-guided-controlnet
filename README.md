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
- [<u>9. Stars, Forked, and Star History</u>](#stars-forked-and-star-history)

<!-- omit in toc -->
# Overview
This is a re-implementation of ControlNet trained with shape masks.
If you have any suggestions about this repo, please feel free to [start a new issue](https://github.com/AlonzoLeeeooo/shape-guided-controlnet/issues/new) or [propose a PR](https://github.com/AlonzoLeeeooo/shape-guided-controlnet/pulls).

[<u><small><🎯Back to Table of Contents></small></u>](#table-of-contents)


<!-- omit in toc -->
# To-Do List
- Regular Maintainence

[<u><small><🎯Back to Table of Contents></small></u>](#table-of-contents)


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

[<u><small><🎯Back to Table of Contents></small></u>](#table-of-contents)


<!-- omit in toc -->
# Implementation Details
The ControlNet model is trained on COCO dataset with 100,000 iterations, along with a batch size of 4.
Each data sample consists of an image, a descriptive caption, and a shape mask.
The image caption directly uses the official annotations (i.e., `captions_train2014.json`) in COCO dataset.
To obtain the shape mask, I select an off-the-shelf saliency detection model [`u2net`](https://github.com/xuebinqin/U-2-Net) to do the automatic annotation for each image.
Model weights of the annotator and the trained ControlNet are released at the [Hugging Face repo]().

[<u><small><🎯Back to Table of Contents></small></u>](#table-of-contents)


<!-- omit in toc -->
# Prerequisites
You can use the one-click installation script `install.sh` to install all the dependencies.

[<u><small><🎯Back to Table of Contents></small></u>](#table-of-contents)


<!-- omit in toc -->
# Training

[<u><small><🎯Back to Table of Contents></small></u>](#table-of-contents)



<!-- omit in toc -->
# Sampling

[<u><small><🎯Back to Table of Contents></small></u>](#table-of-contents)


<!-- omit in toc -->
# Results

[<u><small><🎯Back to Table of Contents></small></u>](#table-of-contents)


<!-- omit in toc -->
# Stars, Forked, and Star History

[<u><small><🎯Back to Table of Contents></small></u>](#table-of-contents)
