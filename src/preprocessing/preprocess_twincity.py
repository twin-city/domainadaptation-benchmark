import mmcv
import os.path as osp
import numpy as np
import pandas as pd
from PIL import Image
import os
from configs.paths_cfg import *
from configs.dataset.dataset_class_matching import *


def convert_img(img, palette, CLASS_MATCHING_TWINCITY_2_CITYSCAPES, threshold=100):
    """
    assign a RGB pixel to a class depending on L1 distance.

    Depends on thresold, ordering of palette.
    Has artefacts on the borders.
    """

    # Image is set apriori to 0
    x_img = np.array(img.convert('RGB'))
    x_img_new = np.zeros_like(x_img)[:, :, 0]

    for class_twincity_id, color in enumerate(palette):
        # image filled with the given color
        image_blank = np.zeros((1080, 1920, 3))
        image_blank[:, :] = color
        # Perform difference, keep under a threshold
        x_dist = np.abs(x_img - image_blank).sum(axis=2)
        # Get the according cityscapes class id
        x_img_new[x_dist < threshold] = CLASS_MATCHING_TWINCITY_2_CITYSCAPES[class_twincity_id]

    return x_img_new


if __name__ == "__main__":
    """
    Segmentation only
    Format the Unreal images to an adapted format for training in MMSegmentation.
    Ad-hoc fix for the semantic extractor bug. Nearby RGB are assigned to a close known RGB value.
    Ordering is ad-hoc.
    """

    # Parameters
    img_dir = 'ColorImage'
    ann_dir = 'SemanticImage'
    new_ann_dir = 'SemanticImage-format-cityscapes'
    threshold = 100  # todo ad-hoc, depends on the semantic extractor

    # define class and palette for better visualization
    class_info = pd.read_csv(osp.join(TWINCITY_ROOT, "../SemanticClasses.csv"), header=None)
    class_info = class_info.set_index(0)
    code = class_info.loc["Road"].values
    classes = list(class_info.index) + ["Person"]
    palette = class_info.values.tolist() + [[199, 4, 158]]  # The pinl for person

    # Get the annotation files to transform
    list_annot = list(mmcv.scandir(osp.join(TWINCITY_ROOT, ann_dir)))
    list_annot.sort()
    list_annot = [x for x in list_annot if "png" in x]

    # For all file in ann_dir, load, transform, and save in new_ann_dir
    for x in list_annot:
        save_path = osp.join(TWINCITY_ROOT, new_ann_dir, x.replace('seg.png',
                                                                  'format_seg.png'))
        # If image not already converted
        if not os.path.exists(save_path):
            print(x)
            img = Image.open(osp.join(TWINCITY_ROOT, ann_dir, x))
            img = convert_img(img, palette, CLASS_MATCHING_TWINCITY_2_CITYSCAPES, threshold=threshold)
            img_rgb = Image.fromarray(img.astype(np.uint8)).convert('P')
            img_rgb.putpalette(np.array(PALETTE_CITYSCAPES, dtype=np.uint8))
            img_rgb.save(save_path)