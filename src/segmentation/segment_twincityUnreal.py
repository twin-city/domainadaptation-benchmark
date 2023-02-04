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
TWINCITYUNREAL_ROOT = "/home/raphael/work/datasets/twincity-Unreal/v2"
img_dir = 'ColorImage'
ann_dir = 'SemanticImage-format-cityscapes'

CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
           'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
           'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
           'bicycle')

PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
           [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
           [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
           [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
           [0, 80, 100], [0, 0, 230], [119, 11, 32]]


# Parameters
max_iter = 1000
num_classes = len(CLASSES)
load_from = osp.join(CHECKPOINT_DIR, 'semanticsegmentation/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth')
#load_from = None
configs_path = 'configs/segmentation/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'
person_weight = 1
work_dir = f'./work_dirs/TwincityUnreal_weight{person_weight}-{max_iter}_loaded{load_from is not None}v2'

# define class and plaette for better visualization
class_info = pd.read_csv(osp.join(TWINCITYUNREAL_ROOT, "../SemanticClasses.csv"), header=None)
class_info = class_info.set_index(0)
code = class_info.loc["Road"].values
classes = list(class_info.index)+["Person"]
palette = class_info.values.tolist()+[[199,   4, 158]] # The pinl for person

#%% Load an image

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

annot_example = "/home/raphael/work/datasets/twincity-Unreal/v2/SemanticImage-format/BasicSequencer.0008format_seg.png"

# Let's take a look at the segmentation map we got
import matplotlib.patches as mpatches
img = Image.open(annot_example)
plt.figure(figsize=(8, 6))
x = np.array(img.convert('RGB'))
im = plt.imshow(np.array(img.convert('RGB')))

# create a patch (proxy artist) for every color
patches = [mpatches.Patch(color=np.array(palette[i])/255.,
                          label=classes[i]) for i in range(len(palette))]
# put those patched as legend-handles into the legend
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
           fontsize='large')
plt.tight_layout()
plt.show()

#%%


#%% Create the train/val on the fly

# split train/val set randomly
split_dir = 'splits'
mmcv.mkdir_or_exist(osp.join(TWINCITYUNREAL_ROOT, split_dir))
filename_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(
    osp.join(TWINCITYUNREAL_ROOT, ann_dir), suffix='format_seg.png')]
with open(osp.join(TWINCITYUNREAL_ROOT, split_dir, 'train.txt'), 'w') as f:
  # select first 4/5 as train set
  train_length = int(len(filename_list)*4/5)
  f.writelines(line.replace("format_seg", "") + '\n' for line in filename_list[:train_length])
with open(osp.join(TWINCITYUNREAL_ROOT, split_dir, 'val.txt'), 'w') as f:
  # select last 1/5 as train set
  f.writelines(line.replace("format_seg", "") + '\n' for line in filename_list[train_length:])

#%% Register new dataset

"""

@DATASETS.register_module()
class TwincityUnrealDataset(CustomDataset):
    CLASSES = classes
    PALETTE = palette

    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='img.jpeg', seg_map_suffix='format_seg.png',
                         split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
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
cfg.data.val.split = 'splits/val.txt'

cfg.data.test.type = cfg.dataset_type
cfg.data.test.data_root = cfg.data_root
cfg.data.test.img_dir = img_dir
cfg.data.test.ann_dir = ann_dir
cfg.data.test.pipeline = cfg.test_pipeline
cfg.data.test.split = 'splits/val.txt'

# We can still use the pre-trained Mask RCNN model though we do not need to
# use the mask branch
cfg.load_from = load_from

# Set up working dir to save files and logs.
cfg.work_dir = work_dir
mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))

cfg.runner.max_iters = max_iter
cfg.log_config.interval = int(max_iter/10)
cfg.evaluation.interval = int(max_iter/2)
cfg.checkpoint_config.interval = int(max_iter/2)

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
#model.decode_head.loss_decode.class_weight = np.ones(shape=(len(palette)))
#model.decode_head.loss_decode.class_weight[-1] = person_weight

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_segmentor(model, datasets, cfg, distributed=False, validate=True,
                meta=dict())

#%% Resume from
# Build the detector
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