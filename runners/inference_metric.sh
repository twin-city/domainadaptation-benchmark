python ../mmdetection/tools/test.py \
    exps/benchv2/pre_train/cocotwincity/cfg.py \
    exps/benchv2/pre_train/cocotwincity/epoch_20.pth \
    --eval bbox \
    --out exps/benchv2/pre_train/cocotwincity/results2.pkl \
    --options "classwise=True"


python ../mmdetection/tools/analysis_tools/confusion_matrix.py exps/benchv2/pre_train/cocotwincity/cfg.py \
  exps/benchv2/pre_train/cocotwincity/results2.pkl \
  exps/benchv2/pre_train/cocotwincity/ \
  --show

#

#    configs/faster_rcnn/faster_rcnn_r50_fpn_1x_cocotwincity.py \