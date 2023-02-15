from mmcv import Config
import mmcv
import os.path as osp
import pandas as pd
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmseg.apis import set_random_seed
from mmseg.utils import get_device
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor
from configs.paths_cfg import *
import os
from mmseg.apis import single_gpu_test
from mmseg.datasets import build_dataloader
from mmcv.cnn.utils import revert_sync_batchnorm
from mmseg.utils import build_dp
from mmcv.runner import load_checkpoint
import numpy as np

def make_cfg_reproducible(cfg):
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)
    cfg.device = get_device()
    return cfg


def perform_inference(output_dir, train_configs_path, checkpoint_path, data_root, img_dir, ann_dir):

    # Output directory create
    mmcv.mkdir_or_exist(os.path.abspath(output_dir))

    #%% cfg
    cfg = Config.fromfile(train_configs_path)
    cfg = make_cfg_reproducible(cfg)
    cfg.dataset_type = dataset_type
    cfg.data_root = data_root
    cfg.data.test.type = dataset_type
    cfg.data.test.data_root = data_root
    cfg.data.test.img_dir = img_dir
    cfg.data.test.ann_dir = ann_dir
    cfg.data.test.pipeline = cfg.test_pipeline  # todo check if we keep ?
    cfg.data.test.split = 'splits/val_3.txt'

    cfg.work_dir = output_dir
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=False,
        workers_per_gpu=1,
        shuffle=False)
    test_loader_cfg = {
        **loader_cfg,
        'samples_per_gpu': 1,
        'shuffle': False,
        **cfg.data.get('test_dataloader', {})
    }

    # build the dataloader from cfg.data.test
    test_dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(test_dataset, **test_loader_cfg)

    #%% Load a model
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    model.CLASSES = test_dataset.CLASSES
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model = revert_sync_batchnorm(model)
    model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

    # Perform inference
    results = single_gpu_test(
        model,
        data_loader,
        False,
        output_dir,
        False,
        0.5,
        pre_eval=["mIoU"])

    # Process the metrics
    metric = test_dataset.evaluate(results, ["mIoU"])
    metric_dict = dict(config=train_configs_path, metric=metric)

    CLASSES = ('road', 'sidewalk', 'building', 'pole', 'vegetation', 'person')

    metric_dict_subset = {f"IoU.{class_name.lower()}": metric_dict["metric"][f"IoU.{class_name.lower()}"] for class_name in
                         CLASSES}
    metric_dict_subset.update({"mIoU": np.mean(list(metric_dict_subset.values()))})

    mmcv.dump(metric_dict, "metrics.json", indent=4)
    mmcv.dump(metric_dict_subset, "metrics_subset.json", indent=4)

    return metric_dict, metric_dict_subset




def get_infos_for_dataset(dataset_type):

    if dataset_type == "TwincityUnrealDataset":
        img_dir = 'ColorImage'
        ann_dir = 'SemanticImage-format-cityscapes'
        data_root = TWINCITYUNREAL_ROOT
    elif dataset_type == "MapillaryVistasDataset":
        img_dir = 'france'
        ann_dir = 'france-formatCityscapes'
        dataset_type = 'MapillaryVistasDataset'
        data_root = MAPILLARY_ROOT
    elif dataset_type == "GTAVDataset":
        img_dir = 'images'
        ann_dir = 'labels'
        data_root = GTAV_ROOT

    dataset_info_dict = {
        "data_root": data_root,
        "img_dir": img_dir,
        "ann_dir": ann_dir}

    return data_root, img_dir, ann_dir



if __name__ == '__main__':

    #todo set paths in config for the pre-trained models ?

    # For a trained Twincity model
    training_dataset_type = "CityscapesDataset"
    checkpoint_path = osp.join(CHECKPOINT_DIR, 'semanticsegmentation/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth')
    train_configs_path = '../mmsegmentation/pspnet_r50-d8_512x1024_40k_cityscapes.py'

    # For a pre-trained Cityscapes model
    training_dataset_type = "TwincityUnrealDataset"
    checkpoint_path = osp.join(CHECKPOINT_DIR, '/home/raphael/work/code/domainadaptation-benchmark/work_dirs/legacy/TwincityUnreal_weight1-1000_loadedTruev2/latest.pth')
    train_configs_path = '../mmsegmentation/pspnet_r50-d8_512x1024_40k_cityscapes.py'


    # Perform inference on twincity & GTAV
    for dataset_type in ["TwincityUnrealDataset", "GTAVDataset"]:
        output_dir = f"../output_segmentation/workdir/{training_dataset_type}-2-{dataset_type}"
        data_infos = get_infos_for_dataset(dataset_type)
        metric_dict, metric_dict_subset = perform_inference(output_dir, train_configs_path, checkpoint_path, *data_infos)
        print(metric_dict_subset)