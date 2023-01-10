import os.path as osp
import json
from configs.paths_cfg import TWINCITY_ROOT


def subsample_coco_json(root_path, json_name, i=100):
    """
    :param root_path: e.g. TWINCITY_ROOT
    :param json_name: e.g. "coco-training.json"
    :param i: e.g. 100
    :return:

    Example :

    root_path = TWINCITY_ROOT
    json_name = "coco-train.json"
    subsample_coco_json(root_path, json_name, i=100)
    """
    print(f"Subsampling at {root_path}/{json_name} to size {i}")

    # load json
    json_path = osp.join(root_path, json_name)
    with open(json_path) as jsonFile:
        coco_json = json.load(jsonFile)

    # do subsampling
    coco_json_subsampled = coco_json.copy()
    coco_json_subsampled["images"] = coco_json_subsampled["images"][:i]
    set_imageid = {x["id"] for x in coco_json_subsampled["images"]}

    # Reset annotations for the subsampled json
    coco_json_subsampled["annotations"] = []

    # Loop all annotations, only append those with seen image id
    for j, annot in enumerate(coco_json["annotations"]):
        if annot["image_id"] in set_imageid:
            coco_json_subsampled["annotations"].append(annot)

    # break # stop at first unseen image id
    # coco_json_subsampled["annotations"] = coco_json_subsampled["annotations"][:j]

    # Save
    with open(osp.join(root_path, f"{json_name.split('.json')[0]}_{i}.json"), 'w') as fh:
        json.dump(coco_json_subsampled, fh)

#%% Subset the datasets

# Subset CARLA dataset
from configs.paths_cfg import CARLA_ROOT
from src.preprocessing.coco_subset import subsample_coco_json
root_path = CARLA_ROOT
json_name = "coco.json"
subsample_coco_json(root_path, json_name, i=50)

# Twincity
root_path = TWINCITY_ROOT
json_name = "coco-train.json"
subsample_coco_json(root_path, json_name, i=50)

