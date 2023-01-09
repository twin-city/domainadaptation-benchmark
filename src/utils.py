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
from mmcv import Config
import numpy as np
from mmdet.apis import set_random_seed
import mmcv
import os
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
import os.path as osp


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


def train_from_config(out_folder, cfg, load_from=None, seed=0,
                      max_epochs=12, use_tensorboard=True,
                      evaluation_interval=5, validate=True):

        # Concatenate Datasets or not
        datasets = [build_dataset([cfg.data.train])]
        cfg.model.roi_head.bbox_head.num_classes = len(datasets[0].CLASSES)
        if load_from is not None:
            cfg.load_from = load_from
        model = build_detector(cfg.model)

        #%% Runner
        cfg.runner.max_epochs = max_epochs
        cfg.evaluation.interval = evaluation_interval
        cfg.checkpoint_config.interval = max_epochs
        cfg.seed = seed
        set_random_seed(seed, deterministic=False)

        #%% CUDA
        cfg.data.workers_per_gpu = 0
        cfg.gpu_ids = range(1)
        cfg.device = 'cuda'

        # %% Logs, working dir to save files and logs.
        if use_tensorboard:
            cfg.log_config.hooks = [
                dict(type='TextLoggerHook'),
                dict(type='TensorboardLoggerHook')]
        else:
            cfg.log_config.hooks = [
                dict(type='TextLoggerHook')]

        cfg.log_config.interval = 20
        cfg.work_dir = f'{out_folder}'
        mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))

        #%% Dump config file
        cfg.dump(osp.join(cfg.work_dir, "cfg.py"))

        #%%
        cfg.evaluation["save_best"] = "bbox_mAP"

        #%% Launch
        train_detector(model, datasets, cfg, distributed=False, validate=validate)