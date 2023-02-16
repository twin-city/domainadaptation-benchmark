
# Cityscapes
CLASSES_CITYSCAPES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
           'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
           'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
           'bicycle')
PALETTE_CITYSCAPES = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
           [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
           [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
           [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
           [0, 80, 100], [0, 0, 230], [119, 11, 32]]
CLASSES_CITYSCAPES_DICT = {class_id: i for i, class_id in enumerate(CLASSES_CITYSCAPES)}

# Twincity
CLASSES_TWINCITY = ('Undefined','Road','Building','Bollard','Tree','Light','Transport', 'Sidewalk', 'Person')

# Matching Twincity to Cityscapes
CLASSES_TWINCITY_2_CITYSCAPES = {
    "Road": 'road',
    "Building": 'building',
    'Bollard': 'pole',
    'Tree': 'vegetation',
    'Light': 'pole',
    'Transport': 'car',
    'Sidewalk': 'sidewalk',
    'Person': 'person',
    'Undefined': 'terrain'
}
CLASS_MATCHING_TWINCITY_2_CITYSCAPES = {class_twincity_id: CLASSES_CITYSCAPES_DICT[CLASSES_TWINCITY_2_CITYSCAPES[class_twincity]]
                                    for class_twincity_id, class_twincity in enumerate(CLASSES_TWINCITY)}


