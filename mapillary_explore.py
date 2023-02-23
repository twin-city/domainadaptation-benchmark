



import json
json_path = "/media/raphael/Projects/datasets/Mapillary-vistasv2/config_v2.0.json"
json_path = "/media/raphael/Projects/datasets/Mapillary-vistasv2/config_v1.2.json"
with open(json_path) as jsonFile:
    mapillary_details = json.load(jsonFile)

classes = [a["name"] for a in mapillary_details["labels"]]
palette = [a["color"] for a in mapillary_details["labels"]]


#%%


from configs.dataset.dataset_class_matching import *

# automatically by name
dict_mapillaryvistas_2_cityscapes = {i: CLASSES_CITYSCAPES.index(x.split("--")[-1]) for i,x in enumerate(classes) if x.split("--")[-1] in CLASSES_CITYSCAPES}

# https://www.researchgate.net/figure/Label-ID-transformations-from-Mapillary-Vistas-to-Cityscapes_tbl2_338116160
dict_completed_by_hand = {
    24: 0,
    41: 0,
    2: 1,
    45: 5,
    47: 5,
    48: 6,
    50: 7,
    20: 12,
    21: 12,
    22: 12,
}
dict_mapillaryvistas_2_cityscapes.update(dict_completed_by_hand)