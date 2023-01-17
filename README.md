


# Domain Adaptation Benchmark

This repository provides benchmark code to assess the performance of a baseline Detection algorithm (Faster-RCNN) 
trained on synthetic data of street-scene, on real-world validation dataset.


| Training Dataset                                | Validation Dataset                   |
|-------------------------------------------------|--------------------------------------|
| - Twincity-unity <br/> - CARLA <br/> - MOTSynth | - PennFudan <br/> - Cityscapes <br/> |


# Installation

See README_install.md

# Dataset Download and preparation

Datasets are converted to coco format for detection.

Available datasets : 
- PennFudan
- MOTSynth (https://github.com/dvl-tum/motsynth-baselines)
- Cityscapes

Not available at the moment : 
- CARLA 
- twincity-unity

Paths in file configs/paths.cfg need to be set at the respective dataset root e.g. 

``MOTSYNTH_ROOT = "../datasets/MOTSynth/"``

# Object Detection

We use MMDetection python library (https://mmdetection.readthedocs.io/en/latest/) to train a Faster-RCNN with a ResNet50 with FPN backbone.

To launch the benchmark you can run src/train_cfg.py (in progress)

# Acknowledgements

We base on several previous works:
- MMDetection library
- MOTSynth
- CARLA

This project is a joint project between Entrepreneur d'Interet General of Etalab, and the Ministry of Interior.