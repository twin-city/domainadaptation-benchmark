from configs.paths_cfg import *
import cv2 as cv
from matplotlib import pyplot as plt
import torch
import numpy as np
import os.path as osp
import time



#%% Load input


img_rgb_path = "/home/raphael/work/datasets/cityscapes/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png"

#%% Segmentation with mmsegmentation
from mmseg.apis import inference_segmentor, init_segmentor
checkpoint_folder_path = "/home/raphael/work/checkpoints/semanticsegmentation/"

config_file = osp.join("configs", 'pspnet_r50-d8_512x512_4x4_20k_coco-stuff10k.py')
checkpoint_file = osp.join(checkpoint_folder_path, 'pspnet_r50-d8_512x512_4x4_20k_coco-stuff10k_20210820_203258-b88df27f.pth')


config_file = osp.join("configs", 'pspnet_r50-d8_512x1024_40k_cityscapes.py')
checkpoint_file = osp.join(checkpoint_folder_path, 'pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth')




model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
time_now = time.time()
result = inference_segmentor(model, img_rgb_path)
time_after_inference = time.time()
inference_time = time_after_inference - time_now
print("first detection time", inference_time)


# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

#%% Inference & time

import time
time_now = time.time()
result = inference_segmentor(model, img_rgb_path)
time_after_inference = time.time()
inference_time = time_after_inference-time_now
print(inference_time)



# Show results
out_folder = "."
model.show_result(img_rgb_path, result,
                  out_file=osp.join(out_folder, 'test_segmentation.jpg'), opacity=0.5)

#%% Train on dataset

from mmcv import Config
from mmseg.datasets import build_dataset
from mmseg.apis import train_segmentor


config_file = osp.join("configs", 'pspnet_r50-d8_512x1024_40k_cityscapes.py')
config_file = osp.join("configs", 'pspnet_r50-d8_512x512_80k_ade20k.py')

cfg = Config.fromfile(config_file)
cfg["data"]["train"]["data_root"] = CITYSCAPES_ROOT
cfg["data"]["train"]["data_root"] = ADE20K_challenge_ROOT
datasets = [build_dataset([cfg.data.train])]

#%%
#import os
#if not osp.exists(cfg["data"]["train"]["data_root"]):
#    os.symlink(CITYSCAPES_ROOT, cfg["data"]["train"]["data_root"])

out_folder = "out/seg"

cfg.gpu_ids = range(1)
cfg.seed = 0
cfg.device = 'cuda'
cfg.work_dir = f'{out_folder}'
# cfg["runner"]["max_epochs"] = 10
cfg["runner"]["max_iters"] = 1000

# cfg["workflow"] = [('train', 2)]
#lr_config = dict(policy='poly', power=0.9, min_lr=0.0001, by_epoch=False)


cfg["lr_config"]["policy"] = "poly"
#%%
#cfg.runner.max_epochs = 200

model = init_segmentor(config_file, device='cuda:0')
train_segmentor(model, datasets, cfg, distributed=False, validate=False)

