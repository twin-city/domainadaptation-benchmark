# dataset settings

from configs.paths_cfg import CARLA_ROOT
from configs.datasets_cfg import train_pipeline, test_pipeline

dataset_type = 'CocoDataset'
classes = ['Person', 'Vehicle', 'Building']

json_file = "coco_50.json"

carla_train = dict(
        pipeline=train_pipeline,
        type='CocoDataset',
        classes=classes,
        ann_file=f'{CARLA_ROOT}/{json_file}',
        img_prefix=f'{CARLA_ROOT}')

carla_val = dict(
        pipeline=test_pipeline,
        type='CocoDataset',
        classes=classes,
        ann_file=f'{CARLA_ROOT}/{json_file}',
        img_prefix=f'{CARLA_ROOT}')

carla_test = dict(
        pipeline=test_pipeline,
        type='CocoDataset',
        classes=classes,
        ann_file=f'{CARLA_ROOT}/{json_file}',
        img_prefix=f'{CARLA_ROOT}')

data = dict(
    train=carla_train,
    val=carla_val,
    test=carla_test,
)


evaluation = dict(interval=1, metric='bbox')