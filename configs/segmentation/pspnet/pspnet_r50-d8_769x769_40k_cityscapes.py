from configs.paths_cfg import MMSEG_DIR

_base_ = [
    f'{MMSEG_DIR}/configs/_base_/models/pspnet_r50-d8.py',
    f'{MMSEG_DIR}/configs/_base_/datasets/cityscapes_769x769.py', f'{MMSEG_DIR}/configs/_base_/default_runtime.py',
    f'{MMSEG_DIR}/configs/_base_/schedules/schedule_40k.py'
]
model = dict(
    decode_head=dict(align_corners=True),
    auxiliary_head=dict(align_corners=True),
    test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513)))
