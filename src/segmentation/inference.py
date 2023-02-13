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

TWINCITYUNREAL_ROOT = "/home/raphael/work/datasets/twincity-Unreal/v2"
img_dir = 'ColorImage'
ann_dir = 'SemanticImage-format-cityscapes'
load_from = osp.join(CHECKPOINT_DIR, 'semanticsegmentation/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth')
configs_path = 'configs/segmentation/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'
configs_path = '../mmsegmentation/pspnet_r50-d8_512x1024_40k_cityscapes.py'
num_classes = 19
max_iter = 10
work_dir = "../output_segmentation/workdir"


#%% CFG à simplifier
cfg = Config.fromfile(configs_path)

# Since we use only one GPU, BN is used instead of SyncBN
cfg.norm_cfg = dict(type='BN', requires_grad=True)
cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
# modify num classes of the model in decode/auxiliary head
cfg.model.decode_head.num_classes = num_classes
cfg.model.auxiliary_head.num_classes = num_classes

# Modify dataset type and path
cfg.dataset_type = 'TwincityUnrealDataset'
cfg.data_root = TWINCITYUNREAL_ROOT

cfg.data.samples_per_gpu = 8
cfg.data.workers_per_gpu = 8

cfg.img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
cfg.crop_size = (256, 256)
cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1920, 1080), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **cfg.img_norm_cfg),
    dict(type='Pad', size=cfg.crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

cfg.test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1920, 1080),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **cfg.img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]



cfg.data.test.type = cfg.dataset_type
cfg.data.test.data_root = cfg.data_root
cfg.data.test.img_dir = img_dir
cfg.data.test.ann_dir = ann_dir
cfg.data.test.pipeline = cfg.test_pipeline
cfg.data.test.split = 'splits/val_3.txt'

# We can still use the pre-trained Mask RCNN model though we do not need to
# use the mask branch
cfg.load_from = load_from

# Set up working dir to save files and logs.
cfg.work_dir = work_dir
mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))

"""
cfg.runner.max_iters = max_iter
cfg.log_config.interval = int(max_iter)
cfg.evaluation.interval = int(max_iter)
cfg.checkpoint_config.interval = int(max_iter)
"""

# Set seed to facitate reproducing the result
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
cfg.device = get_device()



#%%
from mmseg.apis.test import single_gpu_test
cfg_path_cityscapes = "/home/raphael/work/code/domainadaptation-benchmark/configs/pspnet_r50-d8_512x1024_40k_cityscapes.py"
cfg_cityscapes = Config.fromfile(cfg_path_cityscapes)
cfg_cityscapes.data_root = "/home/raphael/work/datasets/cityscapes_v2"
cfg_cityscapes["data"]["test"]["data_root"] = "/home/raphael/work/datasets/cityscapes_v2"
datasets = [build_dataset(cfg_cityscapes.data.test)]



#%%
from mmseg.apis import single_gpu_test
from mmseg.datasets import build_dataloader

data_loader = datasets

distributed = False
loader_cfg = dict(
    # cfg.gpus will be ignored if distributed
    num_gpus=len(cfg.gpu_ids),
    dist=distributed,
    workers_per_gpu=1,
    shuffle=False)
# The overall dataloader settings
loader_cfg.update({
    k: v
    for k, v in cfg.data.items() if k not in [
        'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
        'test_dataloader'
    ]
})
test_loader_cfg = {
    **loader_cfg,
    'samples_per_gpu': 1,
    'shuffle': False,  # Not shuffle by default
    **cfg.data.get('test_dataloader', {})
}
# build the dataloader
data_loader = build_dataloader(build_dataset(cfg.data.test), **test_loader_cfg)

print(next(iter(data_loader)))


#%% Load a model

# ori_shapes = [_['ori_shape'] for _ in next(iter(data_loader))["img_metas"]]

model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
model.CLASSES = datasets[0].CLASSES

#%%
from mmcv.cnn.utils import revert_sync_batchnorm
from mmseg.utils import build_ddp, build_dp, get_device, setup_multi_processes
model = revert_sync_batchnorm(model)
model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

show_dir = "../"

results = single_gpu_test(
    model,
    data_loader,
    pre_eval=["mIoU"],
    show_dir=show_dir)


