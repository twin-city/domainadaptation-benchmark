"""
Depends on .csv output of easysynth
"""


import matplotlib.pyplot as plt
import pandas as pd

from configs.paths_cfg import *
import os.path as osp
from matplotlib.image import imread
import matplotlib.pyplot as plt

#%%
rgb_img_path = osp.join(TWINCITYUNREAL_ROOT, "ColorImage/BasicSequencer.0000.jpeg")
semantic_img_path = osp.join(TWINCITYUNREAL_ROOT, "SemanticImage/BasicSequencer.0000.jpeg")
img_rgb = imread(rgb_img_path)
img_semantic = imread(semantic_img_path)

fig, axes = plt.subplots(1,2)
axes[0].imshow(img_rgb)
axes[1].imshow(img_semantic[:,:,0])
plt.show()


#%%
import pandas as pd
class_info = pd.read_csv(osp.join(TWINCITYUNREAL_ROOT, "../SemanticClasses.csv"), header=None)
class_info = class_info.set_index(0)

code = class_info.loc["Road"].values

import numpy
import numpy as np
road = np.logical_and(img_semantic[:,:,0] == code[0],
                    img_semantic[:,:,1] == code[1])
road = np.logical_and(road, img_semantic[:,:,2] == code[2])
plt.imshow(road)
plt.show()


#%%
import numpy as np

height, width, _ = img_semantic.shape

mylist = []

for x_pos in range(height):
    for y_pos in range(width):
        r,g,b = img_semantic[x_pos, y_pos]
        #mylist.append(img_semantic[x_pos, y_pos])
        mylist.append(r*g*b)

mylist_filtered = [i for i in mylist if i not in [0, 233, 255]]

plt.hist(mylist_filtered)
plt.show()

#%%


