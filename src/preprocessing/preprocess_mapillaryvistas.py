import os.path as osp
import os
import numpy as np

from configs.paths_cfg import MAPILLARY_ROOT
from configs.dataset.dataset_class_matching import *
import json

img_dir = 'france'
ann_dir = 'training/v1.2/labels'


#%% Dict of matching


json_path = f"{MAPILLARY_ROOT}/config_v1.2.json"
with open(json_path) as jsonFile:
    mapillary_details = json.load(jsonFile)
classes = [x["name"] for x in mapillary_details["labels"]]

# automatically by name
dict_mapillaryvistas_2_cityscapes = {i: CLASSES_CITYSCAPES.index(x.split("--")[-1]) for i,x in enumerate(classes) if x.split("--")[-1] in CLASSES_CITYSCAPES}

# https://www.researchgate.net/figure/Label-ID-transformations-from-Mapillary-Vistas-to-Cityscapes_tbl2_338116160
dict_completed_by_hand = {
    24: 0,
    41: 0,
    2: 1,
    45: 5,
    47: 5,
    48: 6,
    50: 7,
    20: 12,
    21: 12,
    22: 12,
}
dict_mapillaryvistas_2_cityscapes.update(dict_completed_by_hand)

#%%

trans_idx = [255]*255

for key,val in dict_mapillaryvistas_2_cityscapes.items():
    trans_idx[key] = val
trans_idx = np.array(trans_idx)


#%%
if not os.path.exists(os.path.join(MAPILLARY_ROOT, "france-formatCityscapes")):
    os.makedirs(os.path.join(MAPILLARY_ROOT, "france-formatCityscapes"))

import cv2
from PIL import Image
for img_name in os.listdir(osp.join(MAPILLARY_ROOT, "france")):
    mask_name = img_name.replace(".jpg", ".png")
    maskpath = osp.join(MAPILLARY_ROOT,"training","v1.2","labels", mask_name)
    mask = np.asarray(Image.open(maskpath))
    mask = trans_idx[mask]
    cv2.imwrite(os.path.join(MAPILLARY_ROOT, "france-formatCityscapes", mask_name), mask.astype(np.uint8))




""" LEGACY for mapillary 2.0


mapillary_class_dict = {
    64: "vegetation",
    61: "sky",
    21: "road",
    24: "sidewalk",
    16: "road", #chemin?
    27: "building",
    16: "wall",
    108: "car",
    45: "passage pieton"
}


#%%

# todo correspondance
trans_idx = [255]*255
trans_idx[64] = 8
trans_idx[21] = 0
trans_idx[27] = 2
trans_idx[108] = 13
trans_idx[45] = 0
trans_idx = np.array(trans_idx)



from PIL import Image

img_path = "/media/raphael/Projects/datasets/Mapillary-vistasv2/training/v2.0/labels/-0C1J9CvgFP4BTVLXNeNZA.png"

img_path = "/media/raphael/Projects/datasets/GTAV/labels/00001.png"
seg = np.array(
    Image.open(osp.join(osp.dirname(__file__), img_path)))

import pandas as pd
print(pd.value_counts(seg.reshape(-1)))

#%%

import matplotlib.pyplot as plt
plt.imshow(255*(seg==45))
plt.show()

"""