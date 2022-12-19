
from benchmark import benchmark
from benchmark_finetuning import benchmark_finetuning
from mmdet.datasets import build_dataset


if __name__ == '__main__':


    """
    Test launch
    """

    """
    pre_train = False
    add_twincity = False
    i = 1
    exp_folder = "exps/launch_test"
    myseed = 0
    classes = ('Window', 'Person', 'Vehicle')
    max_epochs = 10
    """

    """
    ade_size = 64
    benchmark(pre_train, True, i, exp_folder, ade_size, myseed, classes,
              max_epochs=max_epochs,
              log_config_interval=int(ade_size / 64), evaluation_interval=int(max_epochs / 5))
    """

    """
    ade_size = 128
    benchmark(pre_train, add_twincity, i, exp_folder, ade_size, myseed, classes,
              max_epochs=max_epochs,
              log_config_interval=int(ade_size / 64), evaluation_interval=1)

    ade_size = 128
    benchmark(pre_train, add_twincity, i, exp_folder, ade_size, myseed, classes,
              max_epochs=max_epochs,
              log_config_interval=int(ade_size / 64), evaluation_interval=int(max_epochs / 5))
    """

    """
    First Benchmark : What does adding Twin City do to performances on ADE20K of varying size ?
    
    Note that we handle both cases of having seen and unseen classes
    """


    """
    exp_folder = "exps/final/benchmark-c2"
    myseed = 0

    for i in range(1):
        for classes in [('Person', 'Vehicle')]:
            for pre_train in [True]:
                for add_twincity in [True, False]:
                    for ade_size in [512, 2054]:
                        max_epochs = int(15)
                        benchmark(pre_train, add_twincity, i, exp_folder, ade_size, myseed, classes=classes,
                                  max_epochs=max_epochs,
                                  log_config_interval=int(ade_size / 64), evaluation_interval=int(max_epochs / 5))

                    for ade_size in [128]:
                        max_epochs = int(40)
                        benchmark(pre_train, add_twincity, i, exp_folder, ade_size, myseed, classes=classes,
                                  max_epochs=max_epochs,
                                  log_config_interval=int(ade_size / 64), evaluation_interval=int(max_epochs / 5))
    """





    """
    Second Benchmark : Compare pre-training methods
    
    i.e. is twin city a valid pre-trainig method ? When we have all classes at pretraining ? When we don't (eg windows) ?
    
    Later : study for 1 class specific also (eg windows) %TODO TO THINK ABOUT
    """



    max_epochs = 20
    exp_folder = "exps/final/finetuning-c1window"
    myseed = 0

    pretrained_models = {
        # "nopretrain": None,
         "coco": "checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth",
        # "twincity3": "checkpoints/twincity-3class.pth",
        # "twincity1": "checkpoints/twincity-1class(person).pth",
    }

    """
    ade_size = 512
    pretrained_model_name = "coco"
    pretrained_model_path = "checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth"

    """

    # With 1 classes

    classes = ('Window', 'Person', 'Vehicle')
    classes = ['Window']


    for (pretrained_model_name, pretrained_model_path) in pretrained_models.items():
        print(f"=== {pretrained_model_name} ===")
        for ade_size in [2054]:
            print(f"=== {ade_size} ===")
            # max_epochs = int(20 * (2054 / ade_size))
            max_epochs = 15
            benchmark_finetuning(exp_folder, ade_size, classes, pretrained_model_name, pretrained_model_path, myseed,
                                max_epochs=max_epochs,
                                log_config_interval=int(ade_size / 64), evaluation_interval=int(max_epochs / 5))

        for ade_size in [128]:
            print(f"=== {ade_size} ===")
            # max_epochs = int(20 * (2054 / ade_size))
            max_epochs = 40
            benchmark_finetuning(exp_folder, ade_size, classes, pretrained_model_name, pretrained_model_path, myseed,
                                max_epochs=max_epochs,
                                log_config_interval=int(ade_size / 64), evaluation_interval=int(max_epochs / 5))







    #%% Check ADE
    import os.path as osp
    import json
    ade20k_folder = "../../datasets/ADE20K_2021_17_01/"
    ade20k_coco_json_path = osp.join(ade20k_folder, "coco-training_512.json")
    with open(ade20k_coco_json_path) as jsonFile:
        ade20k_coco_json = json.load(jsonFile)
    print(len(ade20k_coco_json["images"]))

    #%%
    from mmcv import Config
    from benchmark import benchmark
    from benchmark_finetuning import benchmark_finetuning
    from mmdet.datasets import build_dataset

    ade_size = 128
    classes = ('Window', 'Person', 'Vehicle')

    # cfg base
    cfg = Config.fromfile('configs/faster_rcnn_r50_fpn_1x_cocotwincityade20kmerged.py')  # Here val is ADE20k

    #  Data
    cfg_data_twincity = Config.fromfile("../synthetic_cv_data_benchmark/datasets/twincity.py")
    cfg_data_ade20k = Config.fromfile("../synthetic_cv_data_benchmark/datasets/ade20k.py")

    # Classes
    if classes is not None:
        # Training
        cfg_data_ade20k.data.train.classes = classes
        cfg_data_twincity.data.train.classes = classes
        cfg.data.train[0].classes = classes
        cfg.data.train[1].classes = classes
        cfg_data_twincity.data.train.classes = classes
        # Validation
        cfg.data.val.classes = classes


    print(build_dataset([cfg_data_twincity.data.train]).datasets)

    for ade_size in [128, 512, 2054]:

        if ade_size != 2054:
            cfg_data_ade20k.data.train.ann_file = f'../../datasets/ADE20K_2021_17_01/coco-training_{ade_size}.json'
        else:
            cfg_data_ade20k.data.train.ann_file = f'../../datasets/ADE20K_2021_17_01/coco-training.json'

        print(build_dataset([cfg_data_ade20k.data.train]).datasets)
