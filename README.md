# Assess the gap between synthetic data and real data

It is presently done by : 
- Performing inference of a pre-trained Cityscapes PSPNet on our synthetic data. \
See results [Cityscapes2Others.md](Cityscapes2Others.md)

![Cityscapes2Twincity](data/Cityscapes2Twincity.jpeg)
*Inference on Twincity example image, from a PSPNet trained on Cityscapes*

In the future we will add :
- Training a PSPNet on our synthetic data and apply it on Cityscapes (& Mapillary Vistas)



# Step-by step

## Installation
Based on mmsegentation, see [README_install.md](README_install.md)

## Dataset Download and preparation

- Set the paths in configs/paths_cfg.py

- Twincity dataset
  - Download
  - Run src/preprocessing/


## Perform inference
- from the root folder : `python src/segmentation/inference.py`




This project is a joint project between Entrepreneur d'Interet General of Etalab, and the Ministry of Interior.