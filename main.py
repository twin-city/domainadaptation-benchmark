

if __name__ == "__main__":
   print("coucou")

   """ Train twincity Validate Cityscapes / Mapillary France
   - Call train segmentor
   - Call test segmentor
   
   Note : 
   - 
   - use only the classes of twincity
   - use both
   """



   #%% Testing


   #%% Single GPU


   #%%

   """
   / home / raphael / work / code / mmsegmentation / tools / test.py
   pspnet_r50 - d8_512x1024_40k_cityscapes_small.py / home / raphael / work / code / mmsegmentation / checkpoints / pspnet_r50 - d8_512x1024_40k_cityscapes_20200605_003338 - 2966598
   c.pth - -show - dir
   out_img / psp_cityscapes_to_cityscapes(pretrained)
   v2 - -eval
   mIoU
   """

   # Load metrics
   import json
   import numpy as np
   json_path = "/home/raphael/work/code/mmsegmentation/work_dirs/pspnet_r50-d8_512x1024_40k_cityscapes_to_twincityunreal/eval_single_scale_20230213_181930.json"

   json_path = "/home/raphael/work/code/mmsegmentation/work_dirs/pspnet_r50-d8_512x1024_40k_cityscapes_to_twincityunreal/eval_single_scale_20230213_184518.json"
   with open(json_path) as jsonFile:
      metrics = json.load(jsonFile)

   # Select subset of metrics
   CLASSES = ('road', 'sidewalk', 'building', 'pole', 'vegetation', 'person')
   class_name = CLASSES[0]
   metrics_selection = {f"IoU.{class_name.lower()}" : metrics["metric"][f"IoU.{class_name.lower()}"] for class_name in CLASSES}
   metrics_selection.update({"mIoU": np.mean(list(metrics_selection.values()))})

   # Print / save
   print(metrics_selection)

   """ Load pre-trained cityscapes Validate twincity
   - Call test segmentor
   """

