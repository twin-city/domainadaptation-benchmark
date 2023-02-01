

#%%

from mmdet.apis import init_detector, inference_detector, train_detector
import mmcv

# Twincity
checkpoint_file = "exps/benchv2/pre_train/cocotwincity/best_bbox_mAP_epoch_10.pth"
#config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_cocotwincity.py'
config_file = "exps/benchv2/pre_train/cocotwincity/cfg.py"

# CARLA
checkpoint_file = "exps/benchv2/pre_train/carla/best_bbox_mAP_epoch_15.pth"
#config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_cocotwincity.py'
config_file = "exps/benchv2/pre_train/carla/cfg.py"



# Cityscapes
checkpoint_file = "exps/benchv2/pre_train/cityscapes/best_bbox_mAP_epoch_5.pth"
#config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_cocotwincity.py'
config_file = "exps/benchv2/pre_train/cityscapes/cfg.py"

# Fudan
checkpoint_file = "exps/benchv2/pre_train/fudan/best_bbox_mAP_epoch_10.pth"
#config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_cocotwincity.py'
config_file = "exps/benchv2/pre_train/fudan/cfg.py"

# MOTSynth
checkpoint_file = "exps/benchv2/pre_train/motsynth/best_bbox_mAP_epoch_15.pth"
#config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_cocotwincity.py'
config_file = "exps/benchv2/pre_train/motsynth/cfg.py"

#%% Load model and cfg
from mmcv import Config
cfg_dict = Config.fromfile(config_file)
model = init_detector(config_file, checkpoint_file, device='cuda:0')
model.CLASSES = cfg_dict["data"]["train"]["classes"]
#%%
import os
import matplotlib.pyplot as plt
plt.rcParams["savefig.bbox"] = "tight"
example_img_path = os.path.join("src", "preprocessing", "example_data")
print(os.listdir(example_img_path))
carla_img = os.path.join(example_img_path, 'CARLA_012081_rgb.jpg')
twincity_img = os.path.join(example_img_path, 'TWINCITYUNITY_rgb_6.png')
motsynth_img = os.path.join(example_img_path, 'MOTSynth_0026.jpg')
motsynth_img = os.path.join(example_img_path, 'motsynth_0013.jpg')
fudan_img = os.path.join(example_img_path, 'FudanPed00054.png')
cityscapes_img = os.path.join(example_img_path, 'CITYSCAPES_munster_000001_000019_leftImg8bit.png')

#%% Imgs


img = motsynth_img



# build the model from a config file and a checkpoint file


# test a single image and show the results
# img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file='result.jpg')
plt.show()