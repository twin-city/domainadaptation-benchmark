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



def benchmark(pre_train, add_twincity, i, exp_folder, ade_size=2054, seed=0, classes=None, max_epochs = 12, use_tensorboard=True,
              evaluation_interval=5, log_config_interval=5):

        #%% cfg base
        cfg = Config.fromfile('configs/faster_rcnn_r50_fpn_1x_cocotwincityade20kmerged.py') # Here val is ADE20k

        #%% Data
        cfg_data_twincity = Config.fromfile("../synthetic_cv_data_benchmark/datasets/twincity.py")
        cfg_data_ade20k = Config.fromfile("../synthetic_cv_data_benchmark/datasets/ade20k.py")

        if ade_size != 2054:
            cfg_data_ade20k.data.train.ann_file = f'../../datasets/ADE20K_2021_17_01/coco-training_{ade_size}.json'
        else:
            cfg_data_ade20k.data.train.ann_file = f'../../datasets/ADE20K_2021_17_01/coco-training.json'

        # Classes
        if classes is not None:
            # Training
            cfg_data_ade20k.data.train.classes = classes
            cfg_data_twincity.data.train.classes = classes
            cfg.data.train[0].classes = classes
            cfg.data.train[1].classes = classes
            cfg_data_twincity.data.train.classes = classes
            # Validation
            cfg.data.val.classes = classes

        # Concatenate Datasets or not
        if add_twincity:
            datasets = [build_dataset([cfg_data_ade20k.data.train, cfg_data_twincity.data.train])]
        else:
            datasets = [build_dataset([cfg_data_ade20k.data.train])]

        #%% Model
        if classes is not None:
            cfg.model.roi_head.bbox_head.num_classes = len(classes)
        else:
            cfg.model.roi_head.bbox_head.num_classes = 3

        # Model
        if pre_train:
            load_from = "checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth"
        else:
            load_from = None
        cfg, model = prepare_cfg_model(cfg, load_from)


        # Runner
        cfg = prepare_cfg_runner(cfg, max_epochs, evaluation_interval, log_config_interval, seed, use_tensorboard)

        # Paths
        if add_twincity:
            add_twincity_str = "+TC"
        else:
            add_twincity_str = ""
        cfg.work_dir = f'{exp_folder}/c{len(classes)}_ade{ade_size}{add_twincity_str}_pretrain{1*pre_train}_it{i}_maxepochs{max_epochs}'
        mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
        cfg.dump(osp.join(cfg.work_dir, "cfg.py"))

        #%% Launch
        train_detector(model, datasets, cfg, distributed=False, validate=True)

"""
if __name__ == '__main__':

    exp_folder = "exps/exp_bench-test"
    pre_train = True
    i = 0
    myseed = 0
    add_twincity = False
    ade_size = 64
    classes = ('Window', 'Person', 'Vehicle')
    benchmark(pre_train, add_twincity, 64, i, exp_folder, myseed, classes, max_epochs=2*3)
    # benchmark(pre_train, add_twincity, 128, i, exp_folder, myseed, classes, max_epochs=3)

    
    for i in range(1):
        for pre_train in [False, True]:
            for classes in [('Window', 'Person', 'Vehicle')]: #('Person', 'Vehicle'),
                for add_twincity in [False, True]:
                    for ade_size in [64, 128, 256]:
                        benchmark(pre_train, add_twincity, ade_size, i, exp_folder, myseed, classes, max_epochs=20)
    """



