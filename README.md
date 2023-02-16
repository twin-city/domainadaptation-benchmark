# Assess the gap between synthetic data and real data

It is presently done by : 
- Performing inference of a pre-trained Cityscapes PSPNet on our synthetic data. \
See results [Cityscapes2Others.md](Cityscapes2Others.md) with \

 `python src/segmentation/inference.py`

![Cityscapes2Twincity](data/Cityscapes2Twincity.jpeg)
*Inference on Twincity example image, from a PSPNet trained on Cityscapes*

In the future we will add :
- Training a PSPNet on our synthetic data and apply it on Cityscapes (& Mapillary Vistas)



# Step-by step

## Installation
Based on mmsegentation, see [README_install.md](README_install.md)

## Dataset Download and preparation



### Twincity dataset
  - Download (contact us directly for the dataset)
  - set TWINCITY_ROOT in configs/paths_cfg
  - Run `python src/preprocessing/prepareTwincity.py`

```
TwincityUnreal
│   SemanticClasses.csv
│
└───v1
│   ...

│
└───v2
   │   file011.txt
   │   file012.txt
   │
   └───ColorImage
   │    │   BasicSequencer.0000img.jpeg
   │    │   BasicSequencer.0001img.jpeg
   │    │   ...
   │ 
   └───SemanticImage  
       │   BasicSequencer.0001seg.jpeg
       │   BasicSequencer.0002seg.jpeg
       │   ...
```

[ColorImage](..%2F..%2Fdatasets%2Ftwincity-Unreal%2Fv2%2FColorImage)
[SemanticImage](..%2F..%2Fdatasets%2Ftwincity-Unreal%2Fv2%2FSemanticImage)

## Perform inference
- from the root folder : `python src/segmentation/inference.py`

--------------------------------------------------------------

--------------------------------------------------------------


# Optional : add other datasets




### (Optional) GTAV dataset
- Download GTAV semantic segmentation dataset
- set GTAV_ROOT in configs/paths_cfg
- Structure should be as follow :

```
GTAV
│
└───images
│   │   00001.png
│   │   00002.png
│   
│   
└───labels
    │   00001.png
    │   00002.png
```

### (Optional) MapillaryVistas, CARLA, 




This project is a joint project between Entrepreneur d'Interet General of Etalab, and the Ministry of Interior.