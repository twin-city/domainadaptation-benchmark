from mmcv import Config
import mmcv
import os.path as osp
from mmseg.apis import set_random_seed
from mmseg.utils import get_device
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from configs.paths_cfg import *
import os
from mmseg.apis import single_gpu_test
from mmseg.datasets import build_dataloader
from mmcv.cnn.utils import revert_sync_batchnorm
from mmseg.utils import build_dp
from mmcv.runner import load_checkpoint
import numpy as np
import json
from src.utils import download_checkpoint

import pandas as pd
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmseg.apis import train_segmentor

def make_cfg_reproducible(cfg):
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)
    cfg.device = get_device()
    return cfg


def perform_inference(output_dir, config_path, checkpoint_path, dataset_type, data_root, img_dir, ann_dir):

    # Output directory create
    mmcv.mkdir_or_exist(os.path.abspath(output_dir))

    #%% cfg
    cfg = Config.fromfile(config_path)

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
    metric_dict = dict(config=config_path, metric=metric)
    mmcv.dump(metric_dict, osp.join(output_dir,"metrics.json"), indent=4)

    return metric_dict




def get_infos_for_dataset(dataset_type):
    if dataset_type == "TwincityDataset":
        img_dir = 'ColorImage'
        ann_dir = 'SemanticImage-format-cityscapes'
        data_root = TWINCITY_ROOT
    elif dataset_type == "MapillaryVistasDataset":
        img_dir = 'france'
        ann_dir = 'france-formatCityscapes'
        dataset_type = 'MapillaryVistasDataset'
        data_root = MAPILLARY_ROOT
    elif dataset_type == "GTAVDataset":
        img_dir = 'images'
        ann_dir = 'labels-format-cityscapes'
        data_root = GTAV_ROOT
    else:
        raise ValueError("Dataset not known")

    return dataset_type, data_root, img_dir, ann_dir



def infer_and_get_metrics(config_path, checkpoint_path, test_datasets_types, output_root_dir):

    cfg = Config.fromfile(config_path)
    training_dataset_type = cfg.data["train"]["type"]
    config_name = config_path.split("/")[-1].split(".py")[0]
    checkpoint_name = checkpoint_path.split("/")[-1]

    # Check that the pre-trained model is present, if not download it
    if not os.path.exists(checkpoint_path):
        checkpoint_dir = "".join(checkpoint_path.split("/")[:-1])
        model_name = checkpoint_name.split("_")[0]
        mmcv.mkdir_or_exist(checkpoint_dir)

        url = f'https://download.openmmlab.com/mmsegmentation/v0.5/{model_name}/{config_name}/{checkpoint_name}'  # noqa

        print("Downloading pre-trained model")
        download_checkpoint(checkpoint_name, model_name, config_name, checkpoint_dir)
        print("Finished downloading pre-trained model")


    # Perform inference on twincity
    for dataset_type in test_datasets_types:
        output_dir = f"{output_root_dir}/{training_dataset_type}-2-{dataset_type}/{config_name}"
        data_infos = get_infos_for_dataset(dataset_type)
        perform_inference(output_dir, config_path, checkpoint_path, *data_infos)

    # Get the results and perform assessments
    results = {}
    for dataset_type in test_datasets_types:
        output_dir = f"{output_root_dir}/{training_dataset_type}-2-{dataset_type}/{config_name}"
        json_path = osp.join(output_dir, "metrics.json")
        with open(json_path) as jsonFile:
            metrics = json.load(jsonFile)
        results[dataset_type] = metrics

    # Compute on a subset
    show_metric = "IoU"
    classes_subset = ('road', 'sidewalk', 'building', 'pole', 'vegetation', 'person')
    metric_class_subset = [f"{show_metric}.{c}" for c in classes_subset]
    for dataset_type in test_datasets_types:
        output_dir = f"{output_root_dir}/{training_dataset_type}-2-{dataset_type}/{config_name}"
        metric_class_value_subset = [results[dataset_type]['metric'][f"{show_metric}.{c}"] for c in classes_subset]
        metric_subset = np.mean(metric_class_value_subset)
        results[dataset_type]['metric'].update({f"m{show_metric}_subset": metric_subset})
        mmcv.dump(results[dataset_type], osp.join(output_dir,"metrics_subset.json"), indent=4)
        #print(results[dataset_type]['metric']['mIoU'])
        #print(results[dataset_type]['metric']['mIoU_subset'])

    # Save results as a .csv, load if existing or create
    import pandas as pd
    csv_results_dir = osp.join("output", "benchmark")
    mmcv.mkdir_or_exist(csv_results_dir)
    csv_results_cityscapes2others = osp.join(csv_results_dir, "Cityscapes-2-Others.csv")
    df_columns_metrics = ["mIoU", "mIoU_subset"]+metric_class_subset
    df_columns = ["train_dataset", "test_dataset"]+["mIoU", "mIoU_subset"]+metric_class_subset+["checkpoint_name"]
    try:
        df = pd.read_csv(csv_results_cityscapes2others)
        df.set_index("setup", inplace=True)
        assert list(df.columns) == df_columns
    except:
        df = pd.DataFrame(columns=df_columns)

    # Set values in dataframe
    for dataset_type in test_datasets_types:
        config_name_df = config_name if config_name != "pspnet_r50-d8_512x1024_40k_cityscapes" else "default"
        row_index = f'{config_name_df}-{dataset_type.replace("Dataset", "")}'
        row_metrics = pd.DataFrame(results[dataset_type]['metric'], index=[row_index])[df_columns_metrics].round(3)
        row_metrics["train_dataset"] = training_dataset_type.replace("Dataset", "")
        row_metrics["test_dataset"] = dataset_type.replace("Dataset", "")
        row_metrics["checkpoint_name"] = checkpoint_name

        if row_index in df.index:
            df = df.drop(row_index)
        df = df.append(row_metrics)

    # Save dataframe
    df.index.name = "setup"
    df.to_csv(csv_results_cityscapes2others)
    with open("Cityscapes2Others.md", 'w') as md:
        df.to_markdown(buf=md, tablefmt="grid")





"""
    # For a pre-trained Cityscapes model
training_dataset_type = "TwincityDataset"
checkpoint_path = osp.join(CHECKPOINT_DIR, '/home/raphael/work/code/domainadaptation-benchmark/work_dirs/legacy/TwincityUnreal_weight1-1000_loadedTruev2/latest.pth')
config_path = '../mmsegmentation/pspnet_r50-d8_512x1024_40k_cityscapes.py'
"""