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
from configs.paths_cfg import CARLA_ROOT




"""
CARLA acquisitions to coco annotations

To extract the bboxes : https://pytorch.org/vision/main/auto_examples/plot_repurposing_annotations.html

For now take only people, and assuming multiplication of G-B is unique per instance

"""

#todo : issue with Vehicle and Building classes : to be investigated

#%% Paths & params
ASSETS_DIRECTORY = "assets"
plt.rcParams["savefig.bbox"] = "tight"
carla_root_path = CARLA_ROOT

#%% Load an image (test), and get height, width

folder_cam12 = "/home/raphael/work/datasets/CARLA/output/fixed_spawn_Town01_v1/cam12"

# Check images
img_rgb_path = os.path.join(folder_cam12, "016281_rgb.jpg")
img_seg_path = os.path.join(folder_cam12, "016282_semanticseg.jpg")
img_instseg_path = os.path.join(folder_cam12, "016283_instanceseg.jpg")

img_rgb = imread(img_rgb_path)
img_seg = imread(img_seg_path)
img_instseg = imread(img_instseg_path)

img_height = img_rgb.shape[0]
img_width = img_rgb.shape[1]
date_captured = None


#%% The bboxes (cf torchvision code)

# todo compute categories
# See https://carla.readthedocs.io/en/latest/ref_sensors/
classes_code = {
    "Person": 4.0,
    "Vehicle": 10.0,
    "Building": 1.0,
}


def get_bbox(img_instseg):
    mask_g = torch.tensor(img_instseg[:,:,1]).unsqueeze(0)
    mask_b = torch.tensor(img_instseg[:,:,2]).unsqueeze(0)
    box_classes = {}

    # filter according to people
    for class_name, class_id in classes_code.items():
        pixels_not_people = np.logical_not(img_instseg[:,:,0]*255 == class_id)
        mask_gb = mask_g * mask_b
        mask_gb[torch.tensor(pixels_not_people).unsqueeze(0)] = 0
        # print(mask_gb.unique())

        # by number of pixels
        #for x in np.unique(img_instseg[:,:,0])*255:
        #    print(x, (img_instseg[:, :, 0]*255 == x).sum())


        # We get the unique colors, as these would be the object ids.
        obj_ids = torch.unique(mask_gb)

        # first id is the background, so remove it.
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set of boolean masks.
        # Note that this snippet would work as well if the masks were float values instead of ints.
        masks = mask_gb == obj_ids[:, None, None]
        boxes = masks_to_boxes(masks)
        box_classes[class_name] = boxes
    return box_classes


#%%

# Convert image to coco

"""
https://arijitray1993.github.io/CARLA_tutorial/
from CARLA docs : https://carla.readthedocs.io/en/latest/tuto_G_bounding_boxes/
"""


simulation_dataset = {
    "info": {
        "maps": ["Town01"],
        "date": "???",
    },

    "licenses": [
    {
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License"
    }],
    "images": [],
    "categories": [
        {"id": 1, "name": "Building", "supercategory": "rdt"},
        {"id": 4, "name": "Person", "supercategory": "rdt"},
        {"id": 10, "name": "Vehicle", "supercategory": "rdt"}],
    "annotations": [],
}




#%%

print("Creating CARLA Dataset")

annotation_id = 0
selected_img_id = []
disgarded_img_id = []

for cam_id in range(40):
    file_paths = os.listdir(os.path.join(carla_root_path, f"cam{cam_id}"))
    rgb_file_paths = [x for x in file_paths if "rgb" in x]

    for img_rgb_path in rgb_file_paths:
        img_id_str = img_rgb_path.split("_rgb")[0]
        img_id = int(img_id_str)
        img_rgb = imread(f"{carla_root_path}/cam{cam_id}/"+img_rgb_path)

        # if img_id == 12041:
        #     print("coucou")

        # todo automatize this
        if cam_id < 3:
            img_instseg_path = f"cam{cam_id}/00{img_id+2}_instanceseg.jpg"
        else:
            img_instseg_path = f"cam{cam_id}/0{img_id+2}_instanceseg.jpg"

        # todo not robust, make it robust + do it at dataset generation
        try:
            img_tensor = torch.tensor(img_rgb[:, :, :3] * 255, dtype=torch.uint8)
            img_tensor = torch.swapaxes(img_tensor, 2, 0)
            img_tensor = torch.swapaxes(img_tensor, 1, 2)
            img_instseg = imread(os.path.join(carla_root_path, img_instseg_path))

            img_bboxes = get_bbox(img_instseg)

            def compute_area(bbox):
                return int((bbox[2]-bbox[0])*(bbox[3]-bbox[1]))



            img_dict = {
                           "license": 1,
                           "file_name": f"cam{cam_id}/"+img_rgb_path,
                           "height": img_height,
                           "width": img_width,
                           "date_captured": date_captured,
                           "id": img_id,
                           "num_pedestrian": len(img_bboxes),
                           "cam_id": cam_id,
                       }

            # Get the bbox (separated by class id)
            annot_img_list = []
            for class_name in classes_code.keys():
                img_class_bboxes = img_bboxes[class_name]
                for bbox in img_class_bboxes:
                    annot_dict = {
                                "id": annotation_id,
                                "segmentation": [],
                                "area": int((bbox[2]-bbox[0])*(bbox[3]-bbox[1])), #TODO compute the area for segmentation instead
                                "iscrowd": 0,
                                "image_id": img_id,
                                "bbox": bbox.numpy().tolist(),
                                "category_id": classes_code[class_name],
                            },
                    annot_img_list.append(annot_dict)
                    annotation_id += 1

            # Decide to append the image or not : criteria is maximal bbox area of car or person
            if max([0] + [x[0]["area"] for x in annot_img_list if x[0]["category_id"] in [4]]) > 250:
                selected_img_id.append(img_id)
                # Append if decided
                simulation_dataset['images'].append(img_dict)
                for annot_img_dict in annot_img_list:
                    simulation_dataset["annotations"].append(annot_img_dict[0]) #TODO arrange this fix
            else:
                disgarded_img_id.append(img_id)

            print(f"Did read : {cam_id} {img_id}")
        except:
            print(f"Could not read : {cam_id} {img_id}")

#%% Save to right format for all
import json
json_path = os.path.join(carla_root_path, "coco.json")

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
drawn_boxes = draw_bounding_boxes(img_tensor, img_bboxes["Vehicle"], colors="red")
show(drawn_boxes)
plt.savefig("src/preprocessing/carla_random_img.jpg")
plt.show()

""" Check segmentation
drawn_masks = []
for mask in img_bboxes["Person"]:
    drawn_masks.append(draw_segmentation_masks(img_tensor, mask, alpha=0.8, colors="blue"))
show(drawn_masks) # 1 is people
plt.show()
"""






#%% Check the bboxes



img_id = np.random.choice(selected_img_id, 1)[0]
img_dict = [x for x in simulation_dataset["images"] if x["id"]==img_id][0]
cam_id = img_dict["cam_id"]
img_id_str = img_rgb_path.split("_rgb")[0]
img_rgb = imread(f"{carla_root_path}/{img_dict['file_name']}")

# if img_id == 12041:
#     print("coucou")

# todo automatize this
if cam_id < 3:
    img_instseg_path = f"cam{cam_id}/00{img_id + 2}_instanceseg.jpg"
else:
    img_instseg_path = f"cam{cam_id}/0{img_id + 2}_instanceseg.jpg"

# todo not robust, make it robust + do it at dataset generation
try:
    img_tensor = torch.tensor(img_rgb[:, :, :3] * 255, dtype=torch.uint8)
    img_tensor = torch.swapaxes(img_tensor, 2, 0)
    img_tensor = torch.swapaxes(img_tensor, 1, 2)
    img_instseg = imread(os.path.join(carla_root_path, img_instseg_path))
    img_bboxes = get_bbox(img_instseg)
except:
    print("could not plot")

import torch
mask_g = torch.tensor(img_instseg[:, :, 1]).unsqueeze(0)
mask_b = torch.tensor(img_instseg[:, :, 2]).unsqueeze(0)
mask_gb = mask_g * mask_b

box_classes = {}


class_name, class_id = 'Vehicle', 4

# filter according to people

pixels_not_people = np.logical_not(img_instseg[:, :, 0] * 255 == class_id)
mask_gb = mask_g * mask_b
mask_gb[torch.tensor(pixels_not_people).unsqueeze(0)] = 0

# print(mask_gb.unique())
# by number of pixels
# for x in np.unique(img_instseg[:,:,0])*255:
#    print(x, (img_instseg[:, :, 0]*255 == x).sum())

# We get the unique colors, as these would be the object ids.
obj_ids = torch.unique(mask_gb)

# first id is the background, so remove it.
obj_ids = obj_ids[1:]

# split the color-encoded mask into a set of boolean masks.
# Note that this snippet would work as well if the masks were float values instead of ints.
masks = mask_gb == obj_ids[:, None, None]
boxes = masks_to_boxes(masks)


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(16,8))
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

from torchvision.utils import draw_bounding_boxes
drawn_boxes = draw_bounding_boxes(img_tensor, boxes, colors="red")
show(drawn_boxes)
plt.show()











#%%
""" For the debugging

from configs.paths_cfg import TWINCITY_ROOT
import json
import os

twincity_path = os.path.join(TWINCITY_ROOT, "coco-val.json")

with open(twincity_path) as jsonFile:
    twincity_dataset = json.load(jsonFile)


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



#%% Twincity example

from configs.paths_cfg import TWINCITY_ROOT
import json
import os

twincity_path = os.path.join(TWINCITY_ROOT, "coco-val.json")

with open(twincity_path) as jsonFile:
    building_json = json.load(jsonFile)
    

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
img_path = os.path.join("src/preprocessing", "FudanPed00054.png")
mask_path = os.path.join("src/preprocessing", "FudanPed00054_mask.png")
img = read_image(img_path)
mask = read_image(mask_path)

"""