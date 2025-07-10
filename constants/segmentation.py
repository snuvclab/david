import json

# paths
COCO_SEG_CONFIG_PTH = "./imports/pointrend/config/pointrend_rcnn_R_50_FPN_3x_coco.yaml"
COCO_SEG_WEIGHTS_PTH = "./imports/pointrend/weights/model_final_edd263.pkl"
COCO_THING_CLASSES_PTH = "constants/coco_thing_classes.json"

LVIS_SEG_CONFIG_PTH = "LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"
LVIS_SEG_WEIGHTS_PTH = "LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"
LVIS_METADATA_PTH = "constants/lvis_metadata.pickle"
LVIS_THING_CLASSES_PTH = "constants/lvis_thing_classes.json"

# coco categories: if it exists, load. if it doesn't exist, create & save.
with open(COCO_THING_CLASSES_PTH, "r") as rf:
    coco_thing_classes = json.load(rf)
COCO_CLASS_ID2NAME = {idx: name for idx, name in enumerate(coco_thing_classes)}
COCO_CLASS_NAME2ID = {v: k for k, v in COCO_CLASS_ID2NAME.items()}

# lvis categories
with open(LVIS_THING_CLASSES_PTH, "r") as rf:
    lvis_thing_classes = json.load(rf)
LVIS_CLASS_ID2NAME = {idx: name for idx, name in enumerate(lvis_thing_classes)}
LVIS_CLASS_NAME2ID = {v: k for k, v in LVIS_CLASS_ID2NAME.items()}
