#!/bin/bash

sh ../mmsegmentation/tools/dist_train.sh configs/pspnet_r50-d8_512x512_80k_ade20k.py 8 --work_dir work_dirs/pspnet_r50-d8_512x512_80k_ade20k/ --deterministic
