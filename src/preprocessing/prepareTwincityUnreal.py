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
TWINCITYUNREAL_ROOT = "/home/raphael/work/datasets/twincity-Unreal/v2"
img_dir = 'ColorImage'
ann_dir = 'SemanticImage'
new_ann_dir = 'SemanticImage-format'

# Parameters
threshold = 100

# define class and plaette for better visualization
class_info = pd.read_csv(osp.join(TWINCITYUNREAL_ROOT, "../SemanticClasses.csv"), header=None)
class_info = class_info.set_index(0)
code = class_info.loc["Road"].values
classes = list(class_info.index)+["Person"]
palette = class_info.values.tolist()+[[199,   4, 158]] # The pinl for person


def convert_img(img, threshold=100):
    """
    assign a RGB pixel to a class depending on L1 distance.

    Depends on thresold, ordering of palette.
    Has artefacts on the borders.
    """

    x_img = np.array(img.convert('RGB'))
    x_img_new = np.zeros_like(x_img)[:,:,0] # Set apriori to 0

    for i, color in enumerate(palette):
        image_blank = np.zeros((1080, 1920, 3))
        image_blank[:, :] = color
        x_dist = np.abs(x_img - image_blank).sum(axis=2)
        plt.imshow(1.0 * (x_dist < threshold))
        x_img_new[x_dist < threshold] = i

    return x_img_new

if __name__ == "__main__":
    list_annot = list(mmcv.scandir(osp.join(TWINCITYUNREAL_ROOT, ann_dir)))
    list_annot.sort()
    list_annot = [x for x in list_annot if "png" in x]

    for x in list_annot:
        save_path = osp.join(TWINCITYUNREAL_ROOT, new_ann_dir, x.replace('seg.png',
                                                                  'format_seg.png'))
        # If image not already converted
        if not os.path.exists(save_path):
            print(x)
            img = Image.open(osp.join(TWINCITYUNREAL_ROOT, ann_dir, x))
            img = convert_img(img, threshold=100)
            img_rgb = Image.fromarray(img.astype(np.uint8)).convert('P')
            img_rgb.putpalette(np.array(palette, dtype=np.uint8))
            img_rgb.save(save_path)