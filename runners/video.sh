python ../mmdetection/demo/video_gpuaccel_demo.py /home/raphael/work/datasets/MOTSynth/MOTSynth_1/004.mp4 \
    exps/benchv2/pre_train/motsynth/cfg.py \
    exps/benchv2/pre_train/motsynth/epoch_20.pth \
    --nvdecode --out result.mp4