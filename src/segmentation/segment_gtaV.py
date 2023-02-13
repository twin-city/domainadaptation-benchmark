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

# paths


CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
           'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
           'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
           'bicycle')

PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
           [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
           [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
           [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
           [0, 80, 100], [0, 0, 230], [119, 11, 32]]

###### Which Dataset ????
dataset_name = "TwincityUnrealDataset"
dataset_root = TWINCITYUNREAL_ROOT
img_dir = 'ColorImage'
ann_dir = 'SemanticImage-format-cityscapes'
###################################################


""" GTAV
img_dir = 'images'
ann_dir = 'labels'
dataset_name = "GTAVDataset"
dataset_root = GTAV_ROOT

"""

# Parameters
max_iter = 2000
load_from = osp.join(CHECKPOINT_DIR, 'semanticsegmentation/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth') # for faster convergence
#load_from = None


#%%
num_classes = len(CLASSES)
configs_path = 'configs/segmentation/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'
work_dir = f'./work_dirs/{dataset_name}-{max_iter}_loaded{load_from is not None}v2'



#%% Train/val done

"""

# split train/val set randomly
split_dir = 'splits'
mmcv.mkdir_or_exist(osp.join(GTAV_ROOT, split_dir))
filename_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(
    osp.join(GTAV_ROOT, ann_dir), suffix='.png')]
with open(osp.join(GTAV_ROOT, split_dir, 'train.txt'), 'w') as f:
  # select first 4/5 as train set
  train_length = int(len(filename_list)*4/5)
  f.writelines(line.replace("format_seg", "") + '\n' for line in filename_list[:train_length])
with open(osp.join(GTAV_ROOT, split_dir, 'val.txt'), 'w') as f:
  # select last 1/5 as train set
  f.writelines(line.replace("format_seg", "") + '\n' for line in filename_list[train_length:])
"""


#%%
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
cfg.dataset_type = dataset_name
cfg.data_root = dataset_root

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

cfg.data.train.type = cfg.dataset_type
cfg.data.train.data_root = cfg.data_root
cfg.data.train.img_dir = img_dir
cfg.data.train.ann_dir = ann_dir
cfg.data.train.pipeline = cfg.train_pipeline
cfg.data.train.split = 'splits/train.txt'

cfg.data.val.type = cfg.dataset_type
cfg.data.val.data_root = cfg.data_root
cfg.data.val.img_dir = img_dir
cfg.data.val.ann_dir = ann_dir
cfg.data.val.pipeline = cfg.test_pipeline
cfg.data.val.split = 'splits/val_small.txt'

cfg.data.test.type = cfg.dataset_type
cfg.data.test.data_root = cfg.data_root
cfg.data.test.img_dir = img_dir
cfg.data.test.ann_dir = ann_dir
cfg.data.test.pipeline = cfg.test_pipeline
cfg.data.test.split = 'splits/val_small.txt'

# We can still use the pre-trained Mask RCNN model though we do not need to
# use the mask branch
cfg.load_from = load_from

# Set up working dir to save files and logs.
cfg.work_dir = work_dir
mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))

cfg.runner.max_iters = max_iter
cfg.log_config.interval = int(max_iter/20)
cfg.evaluation.interval = int(max_iter/5)
cfg.checkpoint_config.interval = int(max_iter/5)

# Set seed to facitate reproducing the result
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
cfg.device = get_device()

cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='TensorboardLoggerHook')]

# Let's have a look at the final config used for training
print(f'Config:\n{cfg.pretty_text}')


#%% Train

# Build the dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_segmentor(cfg.model)
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_segmentor(model, datasets, cfg, distributed=False, validate=True,
                meta=dict())

#%% Resume from


#%% perform an inference on test set

"""

model.cfg = cfg
model.PALETTE = palette

from mmseg.apis import inference_segmentor, init_segmentor

for img_id in np.arange(100,200,10):
    img_path = f"/home/raphael/work/datasets/twincity-Unreal/v2/ColorImage/BasicSequencer.0{img_id}img.jpeg"
    result = inference_segmentor(model, img_path)
    model.show_result(img_path, result, out_file=f"{work_dir}/{img_id}.png", opacity=0.5)
"""