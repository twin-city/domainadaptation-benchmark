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
ann_dir = 'SemanticImage-format'

# Parameters
max_iter = 2000
num_classes = 9
load_from = osp.join(CHECKPOINT_DIR, 'semanticsegmentation/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth')
#load_from = None
configs_path = 'configs/segmentation/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'


person_weight = 1
work_dir = f'./work_dirs/TwincityUnreal_weight{person_weight}-{max_iter}_loaded{load_from is not None}'

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


#%% Register new dataset

@DATASETS.register_module()
class TwincityUnrealDataset(CustomDataset):
    CLASSES = classes
    PALETTE = palette

    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='img.jpeg', seg_map_suffix='format_seg.png',
                         split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None


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
cfg.log_config.interval = 10
cfg.evaluation.interval = 200
cfg.checkpoint_config.interval = 200

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

cfg.load_from = "/home/raphael/work/code/domainadaptation-benchmark/src/segmentation/work_dirs/TwincityUnreal_weight1-2000_loadedFalse/latest.pth"

# Build the dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_segmentor(cfg.model)
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES


#%%

checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
if 'CLASSES' in checkpoint.get('meta', {}):
    model.CLASSES = checkpoint['meta']['CLASSES']

# clean gpu memory when starting a new evaluation.
torch.cuda.empty_cache()

model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)

#%% Resume from
# Build the detector
#%% perform an inference on test set




inference_dir = "src/segmentation/inference"

if not os.path.exists(inference_dir):
    os.makedirs(inference_dir)

model.cfg = cfg
model.PALETTE = palette

from mmseg.apis import inference_segmentor, init_segmentor

for img_id in np.arange(100,200,10):
    img_path = f"/home/raphael/work/datasets/twincity-Unreal/v2/ColorImage/BasicSequencer.0{img_id}img.jpeg"
    result = inference_segmentor(model, img_path)
    model.show_result(img_path, result, out_file=f"{inference_dir}/{img_id}.png", opacity=0.5)