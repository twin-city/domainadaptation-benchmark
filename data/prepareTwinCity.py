from prepare_data.coco_convert import convert_perception
import json
import os

#%%
data_dir = "../../../datasets/twincity-dataset"

# 'Saint-Augustin',
list_of_dirs = ['Saint-Augustin','Belleville','Olympe de gouge','Pantheon','Gambetta']
coco_dict = {}

for neig in list_of_dirs: #'Olympe de gouge','Pantheon']:
    neig_dict = convert_perception(f'{data_dir}/{neig}')
    coco_dict[neig] = neig_dict

def merge_coco(coco_dict, neighboorhoods, filepath):

    concat_coco_dict = {}
    for key in ['info', 'licenses', 'categories']:
        concat_coco_dict[key] = coco_dict["Gambetta"][key]
    concat_coco_dict["images"] = []
    concat_coco_dict["annotations"] = []
    img_count = 0
    annot_count = 0

    for neighboorhood in neighboorhoods:
        print(neighboorhood, len(coco_dict[neighboorhood]["images"]))
        # Append images
        for img0 in coco_dict[neighboorhood]["images"]:
            img = img0.copy()
            img["id"] += img_count
            img["file_name"] = f"{neighboorhood}/{img['filename']}"
            concat_coco_dict["images"].append(img)

        # Translate the img ids in annotations
        coco_annotations = coco_dict[neighboorhood]["annotations"]
        for annot0 in coco_annotations:
            annot = annot0.copy()
            annot["image_id"] += img_count
            annot["id"] += annot_count
            concat_coco_dict["annotations"].append(annot)

        # Add the img count
        img_count += len(coco_dict[neighboorhood]["images"])
        annot_count += len(coco_dict[neighboorhood]["annotations"])

    #save
    with open(filepath, 'w') as fh:
        json.dump(concat_coco_dict, fh)

    return concat_coco_dict


neighboorhoods = ['Saint-Augustin', 'Belleville', 'Olympe de gouge', 'Pantheon', 'Gambetta']
coco = merge_coco(coco_dict,  neighboorhoods, filepath=f'{data_dir}/coco.json')
train_coco = merge_coco(coco_dict,  ['Saint-Augustin', 'Belleville', 'Olympe de gouge'], filepath=f'{data_dir}/coco-train.json')
val_coco = merge_coco(coco_dict,  ['Pantheon'], filepath=f'{data_dir}/coco-val.json')
test_coco = merge_coco(coco_dict,  ['Gambetta'], filepath=f'{data_dir}/coco-test.json')

"""

#%% Load old json coco

data_dir = "../../../datasets/twincity-dataset"
with open(f'{data_dir}/coco-train.json') as jsonFile:
    concat_coco_json = json.load(jsonFile)

from os import path as osp
twincity_folder = "../../../datasets/twincity-dataset/"
coco_json_path = osp.join(twincity_folder, "legacycoco", "coco-train.json")
with open(coco_json_path) as jsonFile:
    twincity_coco_json = json.load(jsonFile)
"""

