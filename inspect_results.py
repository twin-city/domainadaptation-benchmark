from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import os
from os import path as osp
import json
from mmcv import Config
from utils import plot_results, plot_results_ade20k


#%% Load model
exp_folder = "exps/exp_bench-v8"
# classes = ('Window', 'Person', 'Vehicle') # ('Window', 'Person', 'Vehicle')
pre_train = True
i = 0
myseed = 0
classes = ('Window', 'Person', 'Vehicle')
classes = ('Person', 'Vehicle')
ade_size = 512
add_twincity_str = ""
config_file = 'configs/faster_rcnn_r50_fpn_1x_cocotwincityade20kmerged.py'
cfg_dict = Config.fromfile(config_file)
cfg_dict.model.roi_head.bbox_head.num_classes = len(classes)
cfg_dict.data.val.classes = classes
workdir = f'{exp_folder}/c{len(classes)}_ade{ade_size}{add_twincity_str}_pretrain{1*pre_train}_it{i}'
checkpoint_file1 = f'{workdir}/latest.pth'
model1 = init_detector(config_file, checkpoint_file1, cfg_options=cfg_dict)#, device='cuda:0')
model1.CLASSES = classes

#%% Plot results
img, result = plot_results_ade20k(0, model1)

# show_result_pyplot(model1, img, result)

#%% Plot results

