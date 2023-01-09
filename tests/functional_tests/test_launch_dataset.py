from mmcv import Config
from src.utils import train_from_config
from unittest import TestCase

class Test_launch_dataset(TestCase):

    def test_launch_twincityunity(self):

        max_epochs = 1
        myseed = 0

        out_folder = f"test/launch_twincityunity"

        cfg = Config.fromfile('configs/faster_rcnn/faster_rcnn_r50_fpn_1x_cocotwincity.py')
        train_from_config(out_folder, cfg, seed=myseed, max_epochs=max_epochs, use_tensorboard=True)
        return 1

    def test_launch_carla(self):

        max_epochs = 1
        myseed = 0

        out_folder = f"test/launch_carla"

        cfg = Config.fromfile('configs/faster_rcnn/faster_rcnn_r50_fpn_1x_carla.py')
        train_from_config(out_folder, cfg, seed=myseed, max_epochs=max_epochs, use_tensorboard=True)
        return 1

    def test_launch_cityscapes(self):

        max_epochs = 1
        myseed = 0

        out_folder = f"test/launch_cityscapes"

        cfg = Config.fromfile('configs/faster_rcnn/faster_rcnn_r50_fpn_1x_cityscapes.py')
        train_from_config(out_folder, cfg, seed=myseed, max_epochs=max_epochs, use_tensorboard=True)
        return 1