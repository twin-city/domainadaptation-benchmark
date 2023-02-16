"""
Segmentation only
Format the Unreal images to an adapted format for training in MMSegmentation
"""

# Let's take a look at the dataset
import mmcv
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
import pandas as pd
from PIL import Image
import os


# paths
TWINCITY_ROOT = "/home/raphael/work/datasets/twincity-Unreal/v2"
img_dir = 'ColorImage'
ann_dir = 'SemanticImage'
new_ann_dir = 'SemanticImage-format-cityscapes'

# Parameters
threshold = 100

# define class and plaette for better visualization
class_info = pd.read_csv(osp.join(TWINCITY_ROOT, "../SemanticClasses.csv"), header=None)
class_info = class_info.set_index(0)
code = class_info.loc["Road"].values
classes = list(class_info.index)+["Person"]
palette = class_info.values.tolist()+[[199,   4, 158]] # The pinl for person


# Conversion to cityscapes standards
CLASSES_TWINCITY = ('Undefined',
 'Road',
 'Building',
 'Bollard',
 'Tree',
 'Light',
 'Transport',
 'Sidewalk',
 'Person')

CLASSES_CITYSCAPES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
           'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
           'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
           'bicycle')

CLASSES_CITYSCAPES_DICT = {class_id: i for i, class_id in enumerate(CLASSES_CITYSCAPES)}

CLASSES_TWINCITY_2_CITYSCAPES = {
    "Road": 'road',
    "Building": 'building',
    'Bollard': 'pole',
    'Tree': 'vegetation',
    'Light': 'pole',
    'Transport': 'car',
    'Sidewalk': 'sidewalk',
    'Person': 'person',
    'Undefined': 'terrain'
}

CLASSES_TWINCITY_2_CITYSCAPES_ID = {class_twincity_id: CLASSES_CITYSCAPES_DICT[CLASSES_TWINCITY_2_CITYSCAPES[class_twincity]]
                                    for class_twincity_id, class_twincity in enumerate(CLASSES_TWINCITY)}

PALETTE_CITYSCAPES = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
           [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
           [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
           [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
           [0, 80, 100], [0, 0, 230], [119, 11, 32]]



def convert_img(img, threshold=100):
    """
    assign a RGB pixel to a class depending on L1 distance.

    Depends on thresold, ordering of palette.
    Has artefacts on the borders.
    """

    x_img = np.array(img.convert('RGB'))
    x_img_new = np.zeros_like(x_img)[:,:,0] # Set apriori to 0

    for class_twincity_id, color in enumerate(palette):
        # image filled with the given color
        image_blank = np.zeros((1080, 1920, 3))
        image_blank[:, :] = color

        # Perform difference, keep under a threshold
        x_dist = np.abs(x_img - image_blank).sum(axis=2)
        # plt.imshow(1.0 * (x_dist < threshold))

        # Get the according cityscapes class id
        x_img_new[x_dist < threshold] = CLASSES_TWINCITY_2_CITYSCAPES_ID[class_twincity_id]

    return x_img_new


#%%


if __name__ == "__main__":
    list_annot = list(mmcv.scandir(osp.join(TWINCITY_ROOT, ann_dir)))
    list_annot.sort()
    list_annot = [x for x in list_annot if "png" in x]

    for x in list_annot:
        save_path = osp.join(TWINCITY_ROOT, new_ann_dir, x.replace('seg.png',
                                                                  'format_seg.png'))
        # If image not already converted
        if not os.path.exists(save_path):
            print(x)
            img = Image.open(osp.join(TWINCITY_ROOT, ann_dir, x))
            img = convert_img(img, threshold=100)
            img_rgb = Image.fromarray(img.astype(np.uint8)).convert('P')
            img_rgb.putpalette(np.array(PALETTE_CITYSCAPES, dtype=np.uint8))
            img_rgb.save(save_path)