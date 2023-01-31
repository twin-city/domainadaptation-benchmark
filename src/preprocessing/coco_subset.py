import os.path as osp
import json
from configs.paths_cfg import TWINCITY_ROOT, MOTSYNTH_ROOT


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


def trainvaltest_coco_json(root_path, json_name):
    # TODO for CARLA separate per cams ??? For now generic for coco json
    print(f"trainvaltest at {root_path}/{json_name}")

    # load json
    json_path = osp.join(root_path, json_name)
    with open(json_path) as jsonFile:
        coco_json = json.load(jsonFile)

    # do train val test
    n_imgs = len(coco_json['images'])

    coco_train = coco_json.copy()
    coco_val = coco_json.copy()
    coco_test = coco_json.copy()

    # Images
    coco_train["images"] = coco_json["images"][:int(0.6*n_imgs)]
    coco_val["images"] = coco_json["images"][int(0.6*n_imgs):int(0.8*n_imgs)]
    coco_test["images"] = coco_json["images"][int(0.8*n_imgs):]

    # Annotations

    def get_annots(coco_json, target_set_images):
        # Reset annotations for the subsampled json
        annots = []
        # Loop all annotations, only append those with seen image id
        for j, annot in enumerate(coco_json["annotations"]):
            if annot["image_id"] in target_set_images:
                annots.append(annot)
        return annots

    coco_train["annotations"] = get_annots(coco_json, {x["id"] for x in coco_train["images"]})
    coco_val["annotations"] = get_annots(coco_json, {x["id"] for x in coco_val["images"]})
    coco_test["annotations"] = get_annots(coco_json, {x["id"] for x in coco_test["images"]})


    # Save
    with open(osp.join(root_path, f"{json_name.split('.json')[0]}_train.json"), 'w') as fh:
        json.dump(coco_train, fh)
    with open(osp.join(root_path, f"{json_name.split('.json')[0]}_val.json"), 'w') as fh:
        json.dump(coco_val, fh)
    with open(osp.join(root_path, f"{json_name.split('.json')[0]}_test.json"), 'w') as fh:
        json.dump(coco_test, fh)



#%% Train Val Test
from configs.paths_cfg import CARLA_ROOT, PENNFUDANPED_ROOT
from src.preprocessing.coco_subset import subsample_coco_json


#%% Subset the datasets

# Subset CARLA dataset
root_path = CARLA_ROOT
json_name = "coco.json"
trainvaltest_coco_json(root_path, json_name) # not done for carla by hand yet
subsample_coco_json(root_path, json_name, i=50)
# subsample_coco_json(root_path, json_name, i=600)

# Twincity
root_path = TWINCITY_ROOT
json_name = "coco-train.json"
subsample_coco_json(root_path, json_name, i=50)
# subsample_coco_json(root_path, json_name, i=600)

# Fudan
root_path = PENNFUDANPED_ROOT
json_name = "coco.json"
trainvaltest_coco_json(root_path, json_name)


# MoTSynth
import os
root_path = os.path.join(MOTSYNTH_ROOT, "coco annot")
json_name = "004.json"
subsample_coco_json(root_path, json_name, i=600)
trainvaltest_coco_json(root_path, json_name)
