# dataset settings

from configs.paths_cfg import MOTSYNTH_ROOT
from configs.datasets_cfg import train_pipeline, test_pipeline

dataset_type = 'CocoDataset'
classes = ['person']

carla_train = dict(
        pipeline=train_pipeline,
        type='CocoDataset',
        classes=classes,
        ann_file=f'{MOTSYNTH_ROOT}/coco annot/004_train.json',
        img_prefix=f'{MOTSYNTH_ROOT}')

carla_val = dict(
        pipeline=test_pipeline,
        type='CocoDataset',
        classes=classes,
        ann_file=f'{MOTSYNTH_ROOT}/coco annot/004_val.json',
        img_prefix=f'{MOTSYNTH_ROOT}')

carla_test = dict(
        pipeline=test_pipeline,
        type='CocoDataset',
        classes=classes,
        ann_file=f'{MOTSYNTH_ROOT}/coco annot/004_val.json',
        img_prefix=f'{MOTSYNTH_ROOT}')

data = dict(
    train=carla_train,
    val=carla_val,
    test=carla_test,
)


evaluation = dict(interval=1, metric='bbox')