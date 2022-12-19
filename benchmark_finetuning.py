from mmcv import Config
from mmdet.apis import set_random_seed
import mmcv
import os
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
import numpy as np
import os.path as osp
from utils import *




def benchmark_finetuning(exp_folder, ade_size, classes=None, pretrained_model_name="", pretrained_model_path=None, seed=0, max_epochs = 12, use_tensorboard=True,
                         evaluation_interval=5, log_config_interval=5):

        #%% cfg base
        cfg = Config.fromfile('configs/faster_rcnn_r50_fpn_1x_cocotwincityade20kmerged.py') # Here val is ADE20k

        #%% Data
        cfg_data_ade20k = Config.fromfile("../synthetic_cv_data_benchmark/datasets/ade20k.py")
        cfg_data_ade20k.data.train.ann_file = f'../../datasets/ADE20K_2021_17_01/coco-training_{ade_size}.json'

        # Concatenate Datasets or not
        if classes is not None:
                cfg_data_ade20k.data.train.classes = classes
        datasets = [build_dataset([cfg_data_ade20k.data.train])]

        # Model
        load_from = pretrained_model_path
        if classes is not None:
            cfg.model.roi_head.bbox_head.num_classes = len(classes)
        else:
            cfg.model.roi_head.bbox_head.num_classes = 3

        cfg, model = prepare_cfg_model(cfg, load_from)

        # Runner
        cfg = prepare_cfg_runner(cfg, max_epochs, evaluation_interval, log_config_interval, seed, use_tensorboard)

        # Paths
        cfg.work_dir = f'{exp_folder}/ade{ade_size}_pretrain{pretrained_model_name}'
        mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
        cfg.dump(osp.join(cfg.work_dir, "cfg.py"))

        #%% Launch
        train_detector(model, datasets, cfg, distributed=False, validate=True)


"""
if __name__ == '__main__':

    max_epochs = 20
    exp_folder = "exps/benchmark_finetuning-test"
    myseed = 0
    ade_size = 128
    pretrained_model_name = "twincity3"
    pretrained_model_path = "checkpoints/twincity_3classes.pth"

    benchmark_finetuning(exp_folder, ade_size, pretrained_model_name, pretrained_model_path, myseed, max_epochs=max_epochs)

    classes = ('Window', 'Person', 'Vehicle')
    workdir = f'{exp_folder}/ade{ade_size}_pretrain{pretrained_model_name}'

    # img, result = inspect_results(workdir, classes, 20)
"""