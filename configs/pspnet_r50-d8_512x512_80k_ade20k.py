from configs.paths_cfg import MMSEG_DIR

_base_ = [
    f'{MMSEG_DIR}/configs/_base_/models/pspnet_r50-d8.py', f'{MMSEG_DIR}/configs/_base_/datasets/ade20k.py',
    f'{MMSEG_DIR}/configs/_base_/default_runtime.py', f'{MMSEG_DIR}/configs/_base_/schedules/schedule_80k.py'
]
model = dict(
    decode_head=dict(num_classes=150), auxiliary_head=dict(num_classes=150))


log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000)
evaluation = dict(interval=4000, metric='mIoU', pre_eval=True)
