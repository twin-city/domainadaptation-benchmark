# dataset settings
#dataset_type = 'CocoDataset'
#data_root = 'data/coco/'

from configs.paths_cfg import TWINCITY_ROOT

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
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
        img_scale=(1333, 800),
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


classes = ('Window', 'Person', 'Vehicle')

twincity_train = dict(
        pipeline=train_pipeline,
        type='CocoDataset',
        classes=classes,
        ann_file=f'{TWINCITY_ROOT}/coco-train.json',
        img_prefix=f'{TWINCITY_ROOT}')

twincity_val = dict(
        pipeline=test_pipeline,
        type='CocoDataset',
        classes=('Window', 'Person', 'Vehicle'),
        ann_file=f'{TWINCITY_ROOT}/coco-val.json',
        img_prefix=f'{TWINCITY_ROOT}')

twincity_test = dict(
        pipeline=test_pipeline,
        type='CocoDataset',
        classes=('Window', 'Person', 'Vehicle'),
        ann_file=f'{TWINCITY_ROOT}/coco-test.json',
        img_prefix=f'{TWINCITY_ROOT}')

data = dict(
    train=twincity_train,
    val=twincity_val,
    test=twincity_test,
)


evaluation = dict(interval=1, metric='bbox')