# Installation

## Forked mmsegmentation install with conda

`conda create --name openmmlab python=3.8 -y`
`conda activate openmmlab`

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

`pip install -U openmim` \
`mim install mmcv-full`

Clone a forked version of mmsegmentation with custom datasets \
`git clone https://github.com/RaphaelCouronne/mmsegmentation.git` \
`cd mmdetection` \
`pip install -v -e .` 

Add tensorboard for training logs
`conda install tensorboard`

## Possible installation issues
- I had to use a prior version of pytorch/cuda for my setup : \
`conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia`
- Setuptools led to a bug and needed to be downgraded \
`pip install --upgrade setuptools==59.8.0` (see [here](https://stackoverflow.com/questions/71027006/assertionerror-inside-of-ensure-local-distutils-when-building-a-pyinstaller-exe) for details)