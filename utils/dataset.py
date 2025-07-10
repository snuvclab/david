from constants.datasets import ASSETS
from glob import glob

def category2object(dataset, category):
    if dataset == ASSETS.COMASSET:
        object_path = glob(f"data/ComAsset/{category}/*/model.obj")[0]
    elif dataset == ASSETS.BEHAVE:
        object_path = glob(f"data/BEHAVE/objects/{category}/{category}.obj")[0]
    elif dataset == ASSETS.INTERCAP:
        object_path = glob(f"data/INTERCAP/objects/{category}/mesh.obj")[0]
    elif dataset == ASSETS.FULLBODYMANIP:
        object_path = glob(f"data/FullBodyManip/captured_objects/{category}_cleaned_simplified.obj")[0]
    elif dataset == ASSETS.SAPIEN:
        object_path = glob(f"data/SAPIEN/{category}/*/model.obj")[0]
    else: raise ValueError(f"Inappropriate Dataset: {dataset}")
    if len(object_path) == 0: raise ValueError(f"Inappropriate Category: {category}")

    return object_path