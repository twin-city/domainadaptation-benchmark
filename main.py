from src.segmentation.inference import infer_and_get_metrics
import argparse
import os.path as osp

def parse_args():
    parser = argparse.ArgumentParser(description='inference using a trained model')
    parser.add_argument('--config', default="configs/segmentation/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py",
                        help='test config file path',)
    parser.add_argument('--checkpoint_dir', default="checkpoints/",
                        help='Where checkpoints are stored')
    parser.add_argument('--checkpoint_name', default="pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth",
                        help='The pre-trained model. By default the PSPNet trained on Cityscapes.')
    parser.add_argument('--test_datasets', '--list', nargs='+', default=['TwincityDataset'],
                        help='On which dataset do we validate ?')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    checkpoint_path = osp.join(args.checkpoint_dir, args.checkpoint_name)
    infer_and_get_metrics(args.config, checkpoint_path, args.test_datasets)

if __name__ == '__main__':
    """
    By default we use a pre-trained PSPNet on Cityscapes, and perform inference on Twincity.
    """
    main()
