# dataset settings

"""
See https://github.com/alexchungio/mmdetection-demo
"""

from configs.paths_cfg import PENNFUDANPED_ROOT
from configs.datasets_cfg import train_pipeline, test_pipeline

dataset_type = 'CocoDataset'
classes = ['Person']

carla_train = dict(
        pipeline=train_pipeline,
        type=dataset_type,
        classes=classes,
        ann_file=f'{PENNFUDANPED_ROOT}/coco_train.json',
        img_prefix=f'{PENNFUDANPED_ROOT}')

carla_val = dict(
        pipeline=test_pipeline,
        type=dataset_type,
        classes=classes,
        ann_file=f'{PENNFUDANPED_ROOT}/coco_val.json',
        img_prefix=f'{PENNFUDANPED_ROOT}')

carla_test = dict(
        pipeline=test_pipeline,
        type=dataset_type,
        classes=classes,
        ann_file=f'{PENNFUDANPED_ROOT}/coco_test.json',
        img_prefix=f'{PENNFUDANPED_ROOT}')

data = dict(
    train=carla_train,
    val=carla_val,
    test=carla_test,
)


evaluation = dict(interval=1, metric='bbox')


"""

# dataset settings
dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'VOC2007/ImageSets/Main/trainval.txt',
                data_root + 'VOC2012/ImageSets/Main/trainval.txt'
            ],
            img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'],
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')

"""