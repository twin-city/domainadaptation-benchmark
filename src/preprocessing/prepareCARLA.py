


"""
CARLA acquisitions to coco annotations
"""

#%% Paths

folder_cam12 = "/home/raphael/work/datasets/CARLA/output/fixed_spawn_Town01_v1/cam12"

#%% Load an image
import os


img_rgb_path = os.path.join(folder_cam12, "016281_rgb.jpg")
img_seg_path = os.path.join(folder_cam12, "016282_semanticseg.jpg")
img_instseg_path = os.path.join(folder_cam12, "016283_instanceseg.jpg")

import matplotlib

from matplotlib.image import imread

img_rgb = imread(img_rgb_path)
img_seg = imread(img_seg_path)
img_instseg = imread(img_instseg_path)


print(type(img_rgb))

#%%

import matplotlib.pyplot as plt
plt.imshow(img_instseg[:,:,0])
plt.show()

import numpy as np
np.unique(img_instseg) # A lot of different classes in the image



#%%

# sphinx_gallery_thumbnail_path = "../../gallery/assets/repurposing_annotations_thumbnail.png"

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F


ASSETS_DIRECTORY = "assets"

plt.rcParams["savefig.bbox"] = "tight"


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

from torchvision.io import read_image

img_path = os.path.join("src/preprocessing", "FudanPed00054.png")
mask_path = os.path.join("src/preprocessing", "FudanPed00054_mask.png")
img = read_image(img_path)
mask = read_image(mask_path)



#%% Extract bbox
"""
See : https://pytorch.org/vision/main/auto_examples/plot_repurposing_annotations.html
"""


import torchvision.transforms.functional as F

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(16,8))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

import torch

img_tensor = torch.tensor(img_rgb[:,:,:3]*255, dtype=torch.uint8)
img_tensor = torch.swapaxes(img_tensor, 2, 0)
img_tensor = torch.swapaxes(img_tensor, 1, 2)

mask = torch.tensor(img_instseg[:,:,0]).unsqueeze(0)

# We get the unique colors, as these would be the object ids.
obj_ids = torch.unique(mask)

# first id is the background, so remove it.
obj_ids = obj_ids[1:]

# split the color-encoded mask into a set of boolean masks.
# Note that this snippet would work as well if the masks were float values instead of ints.
masks = mask == obj_ids[:, None, None]

#%% Plotted for all people

from torchvision.utils import draw_segmentation_masks

drawn_masks = []
for mask in masks:
    drawn_masks.append(draw_segmentation_masks(img_tensor, mask, alpha=0.8, colors="blue"))

show(drawn_masks[1]) # 1 is people
plt.show()
print("coucou")
# plt.show()

#%% Now for each people separatively
img_instseg = imread(img_instseg_path)



#%% Filter : take only people



#%% Assuming multiplication of G-B is unique per instance



img_tensor = torch.tensor(img_rgb[:,:,:3]*255, dtype=torch.uint8)
img_tensor = torch.swapaxes(img_tensor, 2, 0)
img_tensor = torch.swapaxes(img_tensor, 1, 2)


import torch

def get_bbox(img_instseg, class_id=4.0):
    mask_g = torch.tensor(img_instseg[:,:,1]).unsqueeze(0)
    mask_b = torch.tensor(img_instseg[:,:,2]).unsqueeze(0)
    mask_gb = mask_g * mask_b

    # filter according to people
    pixels_not_people = np.logical_not(img_instseg[:,:,0]*255 == class_id)
    mask_gb[torch.tensor(pixels_not_people).unsqueeze(0)] = 0
    # print(mask_gb.unique())


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


from torchvision.utils import draw_segmentation_masks

drawn_masks = []
for mask in masks:
    print(mask.sum())
    drawn_masks.append(draw_segmentation_masks(img_tensor, mask,
                                               alpha=0.8, colors="red"))

show(drawn_masks) # 1 is people
plt.show()
print("coucou")
# plt.show()


#%% Transform to bboxes

from torchvision.ops import masks_to_boxes

boxes = masks_to_boxes(masks)
print(boxes.size())
print(boxes)

from torchvision.utils import draw_bounding_boxes

drawn_boxes = draw_bounding_boxes(img_tensor, boxes, colors="red")
show(drawn_boxes)
plt.show()


#%%

# Convert image to coco

"""
https://arijitray1993.github.io/CARLA_tutorial/
from CARLA docs : https://carla.readthedocs.io/en/latest/tuto_G_bounding_boxes/
"""


simulation_dataset = {
    "info": {"maps": ["Town01"]},

    "licenses": [
    {
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License"
    }],

    "images": [],
    "categories": [
        {"supercategory": "person", "id": 4, "name": "person" },
    ],
    "annotations": [],
}

#%% For one image

carla_root_path = "/home/raphael/work/datasets/CARLA/output/fixed_spawn_Town01_v1"

img_height = img_rgb.shape[0]
img_width = img_rgb.shape[1]
date_captured = None



# img_path = "cam12/016281_rgb.jpg"
# img_id = int(img_path.split("/")[1].split("_rgb")[0])

#%%

for cam_id in range(40):

    file_paths = os.listdir(os.path.join(carla_root_path, f"cam{cam_id}"))
    rgb_file_paths = [x for x in file_paths if "rgb" in x]

    for img_path in rgb_file_paths:
        img_id_str = img_path.split("_rgb")[0]
        img_id = int(img_id_str)

        #todo automatize this
        if cam_id < 3:
            img_instseg_path = f"cam{cam_id}/00{img_id+2}_instanceseg.jpg"
        else:
            img_instseg_path = f"cam{cam_id}/0{img_id+2}_instanceseg.jpg"

        #todo not robust, make it robust + do it at dataset generation

        try:
            img_tensor = torch.tensor(img_rgb[:, :, :3] * 255, dtype=torch.uint8)
            img_tensor = torch.swapaxes(img_tensor, 2, 0)
            img_tensor = torch.swapaxes(img_tensor, 1, 2)
            img_instseg = imread(os.path.join(carla_root_path, img_instseg_path))
            img_bboxes = get_bbox(img_instseg, 4.0)

            img_dict = {
                           "license": 1,
                           "file_name": f"cam{cam_id}/"+img_path,
                           "height": img_height,
                           "width": img_width,
                           "date_captured": date_captured,
                           "id": img_id,
                           "num_pedestrian": len(img_bboxes),
                       }

            # Get the bbox



            #TODO compute COCO areas ?
            # todo get segmentation masks

            annot_img_list = []
            for bbox in img_bboxes:
                annot_dict = {
                            "segmentation": [],
                            "area": None,
                            "iscrowd": 0,
                            "image_id": img_id,
                            "bbox": bbox.numpy().tolist(),
                            "category_id": 4,
                        },
                annot_img_list.append(annot_dict)

            #todo compute categories

            simulation_dataset['images'].append(img_dict)

            for annot_img_dict in annot_img_list:
                simulation_dataset["annotations"].append(annot_img_dict)

            print(f"Did read : {cam_id} {img_id}")
        except:
            print(f"Cound not read : {cam_id} {img_id}")


#%% Save to right format for all
import json
json_path = os.path.join(carla_root_path, "coco.json")

with open(json_path, 'w') as fh:
    json.dump(simulation_dataset, fh)






#%% Twincity example

from configs.paths_cfg import TWINCITY_ROOT
import json
import os

twincity_path = os.path.join(TWINCITY_ROOT, "coco-val.json")

with open(twincity_path) as jsonFile:
    building_json = json.load(jsonFile)