#_base_ = ['../../mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py', "../datasets/twincity.py"]


_base_ = [
    '../../mmdetection/configs/_base_/models/faster_rcnn_r50_fpn.py',
   # '../../mmdetection/configs/_base_/datasets/coco_detection.py',
    '../datasets/twincity.py',
    '../../mmdetection/configs/_base_/schedules/schedule_1x.py',
    '../../mmdetection/configs/_base_/default_runtime.py'
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


