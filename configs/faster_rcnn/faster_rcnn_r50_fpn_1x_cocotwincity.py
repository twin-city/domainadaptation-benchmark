from configs.paths_cfg import MMDETECTION_ROOT
import os

_base_ = [
    os.path.join(MMDETECTION_ROOT, 'configs/_base_/models/faster_rcnn_r50_fpn.py'),
    '../../configs/datasets/twincity_detection.py',
    os.path.join(MMDETECTION_ROOT, 'configs/_base_/schedules/schedule_1x.py'),
    os.path.join(MMDETECTION_ROOT, 'configs/_base_/default_runtime.py'),
]

model = dict(
    backbone=dict(
        norm_cfg=dict(requires_grad=False),
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')))

# model["roi_head"]["bbox_head"]["num_classes=3"] = 3
# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco/faster_rcnn_r50_fpn_mstrain_3x_coco_20210524_110822-e10bd31c.pth'

# os.path.join(MMDETECTION_ROOT, '/configs/_base_/datasets/coco_detection.py',
#_base_ = [os.path.join(MMDETECTION_ROOT, '/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py', "../datasets/twincity_detection.py"]
