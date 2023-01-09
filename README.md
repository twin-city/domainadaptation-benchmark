# synthetic_cv_data_benchmark
To what extent are synthetic data improving the model ?



# mmdet install

conda create --name openmmlab python=3.8 -y
conda activate openmmlab

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

pip install -U openmim
mim install mmcv-full

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .

conda install tensorboard

# Installation 
- mmdet installations
  - Be careful, it did not install cuda. From pytorch official install : *conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia*
  - 
- install for mmcv (not via conda, rather via pip as advised in website)

# Dataset conversion

### Test on cityscape dataset
- Cityscape dataset conversion
  - launch cityscapes dataset --> folder dataset_converters changed

### ADE20K

python convert.py --input-folder /home/raphael/work/datasets/ADE20K_2021_17_01_nococoyet --output-folder datasets/ADE20K_2021_17_01_coco \
                  --input-format ADE20K --output-format COCO --copy

# Trained models
- mask RCNN : https://github.com/open-mmlab/mmdetection/tree/master/configs/mask_rcnn
- Trained model paths available in configs : https://github.com/open-mmlab/mmdetection/tree/master/configs


# Data
- Description of Cityscape : https://github.com/open-mmlab/mmdetection/blob/master/configs/cityscapes/README.md


# Inference : test on cityscapes

./tools/dist_test.sh configs/cityscapes/mask_rcnn_r50_fpn_1x_cityscapes.py \
    checkpoints/mask_rcnn_r50_fpn_1x_cityscapes_20200227-afe51d5a.pth \
    8  --format-only


../mmdetection/tools/dist_test.sh \
    ../mmdetection/configs/cityscapes/mask_rcnn_r50_fpn_1x_cityscapes.py \
    ../mmdetection/checkpoints/mask_rcnn_r50_fpn_1x_cityscapes_20200227-afe51d5a.pth \
    1  --format-only


./tools/dist_test.sh configs/cityscapes/mask_rcnn_r50_fpn_1x_cityscapes.py \
    checkpoints/mask_rcnn_r50_fpn_1x_cityscapes_20201211_133733-d2858245.pth\
    1  --format-only

# Issues : cityscape : why do we have 8 repeat ?

https://www.cis.upenn.edu/~jshi/ped_html/
