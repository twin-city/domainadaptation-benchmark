from configs.paths_cfg import TWINCITY_ROOT
from configs.datasets_cfg import train_pipeline, test_pipeline

dataset_type = 'CocoDataset'
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
        classes=classes,
        ann_file=f'{TWINCITY_ROOT}/coco-val.json',
        img_prefix=f'{TWINCITY_ROOT}')

twincity_test = dict(
        pipeline=test_pipeline,
        type='CocoDataset',
        classes=classes,
        ann_file=f'{TWINCITY_ROOT}/coco-test.json',
        img_prefix=f'{TWINCITY_ROOT}')

data = dict(
    train=twincity_train,
    val=twincity_val,
    test=twincity_test,
)


evaluation = dict(interval=1, metric='bbox')