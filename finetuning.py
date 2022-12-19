from mmcv import Config
from mmdet.apis import set_random_seed
import mmcv
import os
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
import numpy as np
import os.path as osp

def train_twincity(out_folder, classes, load_from=None, seed=0, max_epochs = 12, use_tensorboard=True):

        #%% cfg base
        cfg = Config.fromfile('configs/faster_rcnn_r50_fpn_1x_cocotwincity.py') # Here val is ADE20k

        #%% Data
        cfg_data_twincity = Config.fromfile("../synthetic_cv_data_benchmark/datasets/twincity.py")

        # Classes
        #
        if classes is not None:
            # Training
            cfg.data.train.classes = classes
            cfg.data.val.classes = classes

        # Concatenate Datasets or not
        datasets = [build_dataset([cfg.data.train])]
        cfg.model.roi_head.bbox_head.num_classes = len(classes)
        if load_from is not None:
            cfg.load_from = load_from
        model = build_detector(cfg.model)

        #%% Runner
        cfg.runner.max_epochs = max_epochs
        cfg.evaluation.interval = 5
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
        print(cfg.data.train.classes)
        print(cfg.data.val.classes)

        cfg.evaluation["save_best"] = "bbox_mAP"

        #%% Launch
        train_detector(model, datasets, cfg, distributed=False, validate=True)


if __name__ == '__main__':

    max_epochs = 20
    out_folder = f"exps/pretrain_twincity/fromscratch-1class-{max_epochs}-v3"
    myseed = 0
    #load_from = "exps/pretrain_twincity/v2/latest.pth"
    classes = ['Person']
    # classes = ('Window', 'Person', 'Vehicle')
    train_twincity(out_folder, classes, seed=myseed, max_epochs=max_epochs, use_tensorboard=True)



    """ PeopleSansPeople
    We set the initial learning rate for all models to 0.02, the initial patience to 38, and the initial epsilon
    to 5. The weight decay is 0.0001, and momentum is 0.9. We perform a linear warm-up period of
    1000 iterations at the start of training (both for training from scratch and transfer learning), where we
    slowly increase the learning rate to the initial learning rate. We use 8 NVIDIA Tesla V100 GPUs
    using synchronized SGD with a mini-batch size of 2 images per GPU. We use the mean pixel value
    and standard deviation from ImageNet for our image normalization in the model. We do not change
    the default augmentations used by Detectron2. We perform the evaluation every two epochs. This
    affects the total number of iterations, the patience period, and learning rate scheduling periods. We
    also fix the model seed to improve reproducibility
    """