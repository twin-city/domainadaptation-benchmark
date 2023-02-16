from mmcv import Config
from unittest import TestCase
from src.segmentation.inference import get_infos_for_dataset, perform_inference
from configs.paths_cfg import CHECKPOINT_DIR, OUT_ROOT
import os.path as osp

class Test_launch_dataset(TestCase):

    def test_inference_from_cityscapes(self):

        mIoU_threshold = 0.1

        # For a trained Twincity model
        training_dataset_type = "CityscapesDataset"
        checkpoint_path = osp.join(CHECKPOINT_DIR,
                                   'semanticsegmentation/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth')
        train_configs_path = 'configs/segmentation/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'
        dataset_type = "TwincityUnrealDataset"

        # Perform inference on twincity
        output_dir = f"{OUT_ROOT}/{training_dataset_type}-2-{dataset_type}"
        data_infos = get_infos_for_dataset(dataset_type)
        metric_dict = perform_inference(output_dir, train_configs_path, checkpoint_path, *data_infos)

        mIoU = metric_dict['metric']["mIoU"]
        self.assertTrue(mIoU > mIoU_threshold)
        print(f"mIoU of pre-trained Cityscapes on twintiy is \n "
              f"------> {mIoU:.2f} \n"
              f"which is above our apriori threshold of \n"
              f"------> {mIoU_threshold}")
        return 1