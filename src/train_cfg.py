from mmcv import Config
from src.utils import train_from_config

if __name__ == '__main__':

    max_epochs = 1
    myseed = 0

    out_folder = f"exps/test"

    cfg = Config.fromfile('configs/faster_rcnn_r50_fpn_1x_cocotwincity.py')
    train_from_config(out_folder, cfg, seed=myseed, max_epochs=max_epochs, use_tensorboard=True)