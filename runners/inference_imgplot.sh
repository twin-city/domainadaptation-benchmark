dataset = "motsynth"

python ../mmdetection/tools/test.py \
    exps/benchv2/pre_train/$(dataset)/cfg.py \
    exps/benchv2/pre_train/$(dataset)/epoch_20.pth \
    --show