from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import os
from os import path as osp
import json
from mmcv import Config
from mmcv import Config
from mmdet.apis import set_random_seed
import mmcv
from mmdet.models import build_detector
import os.path as osp


def plot_results_ade20k(idx_img, model):
    ade20k_folder = "../../datasets/ADE20K_2021_17_01/"
    ade20k_coco_json_path = osp.join(ade20k_folder, "coco-training.json")
    with open(ade20k_coco_json_path) as jsonFile:
        ade20k_coco_json = json.load(jsonFile)

    # Image ade
    ade20k_example = osp.join("../../datasets/ADE20K_2021_17_01/images/ADE/training/urban/street",
                                ade20k_coco_json["images"][idx_img]["file_name"].replace("\\", "/"))
    img_ade20k = mmcv.imread(ade20k_example)
    result_ade20k = inference_detector(model, img_ade20k)

    show_result_pyplot(model, img_ade20k, result_ade20k)

    return img_ade20k, result_ade20k


def plot_results_twincity(idx_img, model):
    twincity_folder = "../../datasets/twincity-dataset/"
    coco_json_path = osp.join(twincity_folder, "coco-train.json")
    with open(coco_json_path) as jsonFile:
        twincity_coco_json = json.load(jsonFile)

    # Image twincity
    twincity_example = osp.join(twincity_folder, twincity_coco_json["images"][idx_img]["file_name"].replace("\\", "/"))
    img_twincity = mmcv.imread(twincity_example)
    result_twincity = inference_detector(model, img_twincity)

    show_result_pyplot(model, img_twincity, result_twincity)

    return img_twincity, result_twincity




def inspect_results(workdir, classes, i=0):

    #%% Set classes
    config_file = osp.join(workdir, "cfg.py")
    cfg_dict = Config.fromfile(config_file)
    cfg_dict.model.roi_head.bbox_head.num_classes = len(classes)
    cfg_dict.data.val.classes = classes

    #%% get latest
    checkpoint_file1 = f'{workdir}/latest.pth'
    model1 = init_detector(config_file, checkpoint_file1, cfg_options=cfg_dict)  # , device='cuda:0')
    model1.CLASSES = classes

    # %% Plot results
    img, result = plot_results_ade20k(i, model1)
    return img, result


#%% Benchmark utils


def prepare_cfg_model(cfg, load_from=None):
    # %% Model
    cfg.load_from = load_from
    model = build_detector(cfg.model)
    return cfg, model


def prepare_cfg_runner(cfg, max_epochs, evaluation_interval, log_config_interval, seed, use_tensorboard):
    # %% Runner
    cfg.runner.max_epochs = max_epochs
    cfg.evaluation.interval = evaluation_interval
    cfg.log_config.interval = log_config_interval
    cfg.checkpoint_config.interval = max_epochs
    cfg.seed = seed
    set_random_seed(seed, deterministic=False)

    # %% CUDA
    cfg.data.workers_per_gpu = 0
    cfg.gpu_ids = range(1)
    cfg.device = 'cuda'
    cfg.optimizer.lr = 0.02 / 8

    # %% Logs, working dir to save files and logs.
    cfg.evaluation["save_best"] = "bbox_mAP"
    if use_tensorboard:
        cfg.log_config.hooks = [
            dict(type='TextLoggerHook'),
            dict(type='TensorboardLoggerHook')]
    else:
        cfg.log_config.hooks = [
            dict(type='TextLoggerHook')]
    return cfg