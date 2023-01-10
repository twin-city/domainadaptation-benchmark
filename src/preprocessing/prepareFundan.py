from torchvision.ops import masks_to_boxes
import os
from matplotlib.image import imread
import matplotlib
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torchvision.io import read_image
from torchvision.utils import draw_segmentation_masks
import torch
from configs.paths_cfg import PENNFUDANPED_ROOT


"""
Fundan to coco annotations
https://github.com/omarfoq/Pedestrian_Detection/blob/master/utils/dataset.py

Or try Pascal 2 Coco : https://blog.roboflow.com/how-to-convert-annotations-from-voc-xml-to-coco-json/
"""

#todo : issue with Vehicle and Building classes : to be investigated

#%% Paths & params
ASSETS_DIRECTORY = "assets"
plt.rcParams["savefig.bbox"] = "tight"

#%% Load an image (test), and get height, width

folder = PENNFUDANPED_ROOT

# Check images
img_rgb_path = os.path.join(folder, "PNGImages", "FudanPed00001.png")
#img_seg_path = os.path.join(folder_, "016282_semanticseg.jpg")
img_instseg_path = os.path.join(folder, "PedMasks", "FudanPed00014_mask.png")

img_rgb = imread(img_rgb_path)
#img_seg = imread(img_seg_path)
img_instseg = imread(img_instseg_path)

img_height = img_rgb.shape[0]
img_width = img_rgb.shape[1]
date_captured = None





def get_bbox(img_instseg):
    """
    Be careful, different id from CARLA.

    Here id is the first !

    :param img_instseg:
    :return:
    """

    mask_gb = torch.tensor(img_instseg)

    # We get the unique colors, as these would be the object ids.
    obj_ids = torch.unique(mask_gb)

    # first id is the background, so remove it.
    obj_ids = obj_ids[1:]

    # split the color-encoded mask into a set of boolean masks.
    # Note that this snippet would work as well if the masks were float values instead of ints.
    masks = mask_gb == obj_ids[:, None, None]
    boxes = masks_to_boxes(masks)

    return boxes


#%%

# Convert image to coco

"""
https://arijitray1993.github.io/CARLA_tutorial/
from CARLA docs : https://carla.readthedocs.io/en/latest/tuto_G_bounding_boxes/
"""


simulation_dataset = {
    "info": {
        "name": "Fudan",
        "date": "2004",
    },

    "licenses": [],
    "images": [],
    "categories": [
        {"id": 1, "name": "Person", "supercategory": "rdt"},
    ],
    "annotations": [],
}




#%%

print("Creating Fudan Dataset")

annotation_id = 0


rgb_file_paths = os.listdir(os.path.join(PENNFUDANPED_ROOT, "PNGImages"))

for img_rgb_path in rgb_file_paths:
    img_id_str = img_rgb_path.split("Ped")[1].split(".png")[0]
    img_id = int(img_id_str)
    img_rgb = imread(f"{PENNFUDANPED_ROOT}/PNGImages/{img_rgb_path}")

    img_instseg_path = img_rgb_path.split(".png")[0]+"_mask"+".png"

    # todo not robust, make it robust + do it at dataset generation

    img_tensor = torch.tensor(img_rgb[:, :, :3] * 255, dtype=torch.uint8)
    img_tensor = torch.swapaxes(img_tensor, 2, 0)
    img_tensor = torch.swapaxes(img_tensor, 1, 2)
    img_instseg = imread(os.path.join(PENNFUDANPED_ROOT, "PedMasks", img_instseg_path))
    img_bboxes = get_bbox(img_instseg)

    img_dict = {
                   "license": 1,
                   "file_name": f""+img_rgb_path,
                   "height": img_height,
                   "width": img_width,
                   "date_captured": date_captured,
                   "id": img_id,
                   "num_pedestrian": len(img_bboxes),
               }

    # Get the bbox (separated by class id)
    annot_img_list = []



    img_class_bboxes = img_bboxes
    for bbox in img_class_bboxes:
        annot_dict = {
                    "id": annotation_id,
                    "segmentation": [],
                    "area": int((bbox[2]-bbox[0])*(bbox[3]-bbox[1])), #TODO compute the area for segmentation instead
                    "iscrowd": 0,
                    "image_id": img_id,
                    "bbox": bbox.numpy().tolist(),
                    "category_id": "Person",
                },
        annot_img_list.append(annot_dict)
        annotation_id += 1


    simulation_dataset['images'].append(img_dict)
    for annot_img_dict in annot_img_list:
        simulation_dataset["annotations"].append(annot_img_dict[0]) #TODO arrange this fix
    print(f"Did read : {img_id}")
    #except:
    #    print(f"Could not read : {img_id}")

#%% Save to right format for all
import json
json_path = os.path.join(PENNFUDANPED_ROOT, "legacy/coco.json")

with open(json_path, 'w') as fh:
    json.dump(simulation_dataset, fh)




#%% Check if instance seg was okay

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(16,8))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])



# Check bboxes
from torchvision.utils import draw_bounding_boxes
drawn_boxes = draw_bounding_boxes(img_tensor, img_bboxes, colors="red")
show(drawn_boxes)
plt.savefig("src/preprocessing/fudan_last_img.jpg")
