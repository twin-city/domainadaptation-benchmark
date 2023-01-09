import os.path as osp
import json




#%%


def subsample_ade20k(ade20k_folder, i):

    print(f"Subsampling ADE20K to size {i}")

    # load json
    ade20k_coco_json_path = osp.join(ade20k_folder, "coco-training.json")
    with open(ade20k_coco_json_path) as jsonFile:
        ade20k_coco_json = json.load(jsonFile)

    #do subsampling
    ade20k_coco = ade20k_coco_json.copy()
    ade20k_coco["images"] = ade20k_coco["images"][:i]
    for j, annot in enumerate(ade20k_coco["annotations"]):
        if annot["image_id"] >= i:
            break
    ade20k_coco["annotations"] = ade20k_coco["annotations"][:j]
    
    with open(osp.join(ade20k_folder, f"coco-training_{i}.json"), 'w') as fh:
        json.dump(ade20k_coco, fh)

    """
    from mmcv import Config
    from mmdet.datasets import build_dataset
    ade_size = 512
    cfg_data_ade20k = Config.fromfile("../../synthetic_cv_data_benchmark/datasets/ade20k.py")
    cfg_data_ade20k.data.train.ann_file = f'../../../datasets/ADE20K_2021_17_01/coco-training_{ade_size}.json'
    datasets = [build_dataset([cfg_data_ade20k.data.train])]
    datasets[0].datasets
    """

ade20k_folder = "../../../datasets/ADE20K_2021_17_01/"

for i in [64, 128, 256, 512, 1024, 2054]:
    subsample_ade20k(ade20k_folder, i)