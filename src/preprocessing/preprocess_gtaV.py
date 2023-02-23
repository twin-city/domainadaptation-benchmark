import os.path as osp
import os

import mmcv
import numpy as np
from PIL import Image

"""
from https://bitbucket.org/visinf/projects-2016-playing-for-data/src/master/label/initLabels.m
"""



#%%

CLASSES_GTAV = [
    'unlabeled','ego vehicle','rectification border','out of roi','static',
    'dynamic','ground','road' ,'sidewalk','parking',
    'rail track','building', 'wall','fence' ,'guard rail',
    'bridge','tunnel','pole','polegroup','traffic light',
    'traffic sign', 'vegetation','terrain','sky' ,'person',
    'rider', 'car','truck','bus' ,'caravan', 
    'trailer','train', 'motorcycle', 'bicycle','license plate']

PALETTE_GTAV = [
  [0,  0,  0],
  [0,  0,  0],
  [0,  0,  0],
  [0,  0,  0],
[20,  20,  20],
[111, 74,  0],
    [ 81,  0, 81],
[128, 64,128],
[244, 35,232],
[250,170,160],
[230,150,140],
    [ 70, 70, 70],
[102,102,156],
[190,153,153],
[180,165,180],
[150,100,100],
[150,120, 90],
[153,153,153],
[153,153,153],
[250,170, 30],
[220,220,  0],
[107,142, 35],
[152,251,152],
    [ 70,130,180],
[220, 20, 60],
[255,  0,  0],
    [  0,  0,142],
  [0,  0, 70],
  [0, 60,100],
  [0,  0, 90],
  [0,  0,110],
  [0, 80,100],
  [0,  0,230],
[119, 11, 32],
    [  0,  0,142],
]

labelClasses = [5,6,7,8,9,12,13,14,15,16,17,18,20,21,22,
                23,24,25,26,27,28,29,31,32,33,34,35]

from configs.dataset.dataset_class_matching import *
#%% Which are not present in cityscapes ??? For now we don't handle themp
missing_labels = [CLASSES_GTAV[i-1] for i in labelClasses if CLASSES_GTAV[i-1] not in CLASSES_CITYSCAPES]

#%%
classes_id2name = {i: x for i,x in enumerate(CLASSES_GTAV)}
classes_name2id = {x: i for i,x in enumerate(CLASSES_GTAV)}


#%% All Cityscapes labels are found in GTAV

dict_GTAV_2_Cityscapes = {i: CLASSES_CITYSCAPES.index(class_name) for i, class_name in enumerate(CLASSES_GTAV) if class_name in CLASSES_CITYSCAPES}
dict_Cityscapes_2_GTAV = {i: CLASSES_GTAV.index(class_name)  for i, class_name in enumerate(CLASSES_CITYSCAPES)}


#%%

from configs.paths_cfg import *
out_dir = "/home/raphael/work/datasets/GTAV_local"

ann_dir = "labels"
new_ann_dir = "labels-format-cityscapes"

mmcv.mkdir_or_exist(osp.join(out_dir, new_ann_dir))


for i in range(1, 10):
    label_path = osp.join(GTAV_ROOT, ann_dir, f"0000{i}.png")
    save_path = osp.join(out_dir, new_ann_dir, f"0000{i}.png")
    img = Image.open(label_path)
    x_img = np.array(img)
    for key, val in dict_GTAV_2_Cityscapes.items():
        x_img[x_img == key] = val

    img_rgb = Image.fromarray(x_img.astype(np.uint8)).convert('P')
    img_rgb.putpalette(np.array(PALETTE_CITYSCAPES, dtype=np.uint8))
    img_rgb.save(save_path)


""" Eventually plot it
#%%
img_path = "/media/raphael/Projects/datasets/GTAV/labels/00001.png"



#%%
import matplotlib.pyplot as plt
img_path = "/media/raphael/Projects/datasets/GTAV/images/00003.png"

seg = np.array(
    Image.open(osp.join(osp.dirname(__file__), img_path)))

plt.imshow(seg)
plt.show()

#%%
img_path = "/media/raphael/Projects/datasets/GTAV/labels/00003.png"

seg = np.array(
    Image.open(osp.join(osp.dirname(__file__), img_path)))

import pandas as pd
print(pd.value_counts(seg.reshape(-1)))

import matplotlib.pyplot as plt
plt.imshow(255*(seg==7))
plt.show()
"""