from mmcv import Config
from src.utils import train_from_config

if __name__ == '__main__':

    max_epochs = 20
    myseed = 0

    """
    dataset = "carla"
    cfg = Config.fromfile(f'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_{dataset}.py')

    train_from_config(out_folder, cfg, seed=myseed, max_epochs=max_epochs, use_tensorboard=True)
    """

    for dataset in ["carla", "motsynth", "fudan", "cocotwincity", "cityscapes"]:
        cfg = Config.fromfile(f'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_{dataset}.py')
        out_folder = f"exps/benchv2/pre_train/{dataset}"
        train_from_config(out_folder, cfg, seed=myseed, max_epochs=max_epochs, use_tensorboard=True)