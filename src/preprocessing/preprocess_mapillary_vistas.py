from mmcv import Config
import mmcv
import os.path as osp
import pandas as pd
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmseg.apis import set_random_seed
from mmseg.utils import get_device
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor
from configs.paths_cfg import *
import os
from mmseg.apis import single_gpu_test
from mmseg.datasets import build_dataloader
from mmcv.cnn.utils import revert_sync_batchnorm
from mmseg.utils import build_dp
from mmcv.runner import load_checkpoint
import numpy as np

MAPILLARY_ROOT = "/media/raphael/Projects/datasets/Mapillary-vistasv2"
img_dir = 'france'
ann_dir = 'training/v2.0/labels'

#%%

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

#%%



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

"""
64     2925507
21     2257427
24      787342
16      608745
27      494888
12      382818
122     170729
36      141618
4       100076
"""

#%%

# todo correspondance
trans_idx = [255]*255
trans_idx[64] = 8
trans_idx[21] = 0
trans_idx[27] = 2
trans_idx[108] = 13
trans_idx[45] = 0
trans_idx = np.array(trans_idx)

#%%


#%%
if not os.path.exists(os.path.join(MAPILLARY_ROOT, "france-formatCityscapes")):
    os.makedirs(os.path.join(MAPILLARY_ROOT, "france-formatCityscapes"))

import cv2
from PIL import Image
for img_name in os.listdir(osp.join(MAPILLARY_ROOT, "france")):
    mask_name = img_name.replace(".jpg", ".png")
    maskpath = osp.join(MAPILLARY_ROOT,"training","v2.0","labels", mask_name)
    mask = np.asarray(Image.open(maskpath))
    mask = trans_idx[mask]
    cv2.imwrite(os.path.join(MAPILLARY_ROOT, "france-formatCityscapes", mask_name), mask.astype(np.uint8))
