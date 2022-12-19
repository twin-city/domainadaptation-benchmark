# dataset settings
#dataset_type = 'CocoDataset'
#data_root = 'data/coco/'

import os


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
#img_norm_cfg = dict(
#    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

twincity_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
twincity_test_pipeline = [
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


#%% Twin City

twincity_train = dict(
        pipeline=twincity_train_pipeline,
        type='CocoDataset',
        classes=classes,
        ann_file='../../datasets/twincity-dataset/coco-train.json',
        img_prefix='../../datasets/twincity-dataset')

twincity_val = dict(
        pipeline=twincity_test_pipeline,
        type='CocoDataset',
        classes=('Window', 'Person', 'Vehicle'),
        ann_file='../../datasets/twincity-dataset/coco-val.json',
        img_prefix='../../datasets/twincity-dataset')

twincity_test = dict(
        pipeline=twincity_test_pipeline,
        type='CocoDataset',
        classes=('Window', 'Person', 'Vehicle'),
        ann_file='../../datasets/twincity-dataset/coco-test.json',
        img_prefix='../../datasets/twincity-dataset')

#%% ADE20K

"""

ade20k_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]


ade20k_val_pipeline = [
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
"""

ade20k_val_pipeline = twincity_test_pipeline
ade20k_train_pipeline = twincity_train_pipeline

ade20k_folder = "../../datasets/ADE20K_2021_17_01/"
img_prefix_training = "../../datasets/ADE20K_2021_17_01/images/ADE/training/urban/street"
img_prefix_validation = "../../datasets/ADE20K_2021_17_01/images/ADE/validation/urban/street"

ade_train = dict(
        pipeline=ade20k_train_pipeline,
        type='CocoDataset',
        classes=('Window', 'Person', 'Vehicle'),
        ann_file=os.path.join(ade20k_folder, "coco-training.json"),
        img_prefix=img_prefix_training,
    )

ade_train_16 = dict(
        pipeline=ade20k_train_pipeline,
        type='CocoDataset',
        classes=('Window', 'Person', 'Vehicle'),
        ann_file=os.path.join(ade20k_folder, "coco-training_16.json"),
        img_prefix=img_prefix_training,
    )

ade_val = dict(
        pipeline=ade20k_val_pipeline,
        type='CocoDataset',
        classes=('Window', 'Person', 'Vehicle'),
        ann_file=os.path.join(ade20k_folder, "coco-validation.json"),
        img_prefix=img_prefix_validation,
    )


ade_train_64 = dict(
        pipeline=ade20k_train_pipeline,
        type='CocoDataset',
        classes=('Window', 'Person', 'Vehicle'),
        ann_file=os.path.join(ade20k_folder, "coco-training_64.json"),
        img_prefix=img_prefix_training,
    )

ade_train_256 = dict(
        pipeline=ade20k_train_pipeline,
        type='CocoDataset',
        classes=('Window', 'Person', 'Vehicle'),
        ann_file=os.path.join(ade20k_folder, "coco-training_256.json"),
        img_prefix=img_prefix_training,
    )

ade_train_1024 = dict(
        pipeline=ade20k_train_pipeline,
        type='CocoDataset',
        classes=('Window', 'Person', 'Vehicle'),
        ann_file=os.path.join(ade20k_folder, "coco-training_1024.json"),
        img_prefix=img_prefix_training,
    )



merged_16 = [twincity_train, ade_train_16]
merged_64 = [twincity_train, ade_train_64]
merged_256 = [twincity_train, ade_train_256]
merged_1024 = [twincity_train, ade_train_1024]

data = dict(
    train=[
        twincity_train,
        ade_train_16
    ],
    val=ade_val,
    test=ade_val
)


evaluation = dict(interval=1, metric='bbox')

