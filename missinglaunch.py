
from benchmark import benchmark
from benchmark_finetuning import benchmark_finetuning
from mmdet.datasets import build_dataset


if __name__ == '__main__':

    add_twincity = False
    i = 0
    exp_folder = "exps/missing"
    myseed = 0
    classes = ('Window', 'Person', 'Vehicle')
    max_epochs = 40
    ade_size = 128

    for pre_train in [True, False]:

        benchmark(pre_train, add_twincity, i, exp_folder, ade_size, myseed, classes=classes,
                            max_epochs=max_epochs,
                            log_config_interval=int(ade_size / 64), evaluation_interval=int(max_epochs / 5))

    pretrained_model_name = "twincity3"
    pretrained_model_path = "checkpoints/twincity-3class.pth"
    benchmark_finetuning(exp_folder, ade_size, classes, pretrained_model_name, pretrained_model_path, myseed,
                         max_epochs=max_epochs,
                         log_config_interval=int(ade_size / 64), evaluation_interval=int(max_epochs / 5))