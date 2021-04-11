import torch, torchvision
import detectron2
from pycococreatortools import binary_mask_to_polygon
from detectron2.utils.logger import setup_logger
from detectron2.structures import BoxMode

import os
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import math
import glob
import pycocotools
#from google.colab.patches import cv2_imshow
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from tqdm import tqdm


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg


def gen_masks (path):
    intmask = cv2.imread(path)
    masklist = []
    n = np.max(intmask)
    for i in range(1, n):
        copy = np.copy(intmask[:,:,0])
        copy[copy < i] = 0
        copy[copy > i] = 0
        copy[copy > 0] = 1
        masklist.append(np.array(copy, dtype = np.uint8))
    return masklist

def get_data_dicts(mode):
    categories = os.listdir('../data/alex/animal_database')
    paths = {}
    for each in categories:
        paths[each] = {'originals':'', 'masks':''}
        paths[each]['originals'] = sorted(glob.glob(os.getenv("DATA_ALEX")+f'/animal_database/{each}/original/*'))
        paths[each]['masks'] = sorted(glob.glob(os.getenv("DATA_ALEX")+f'/animal_database/{each}/segment/*'))

    dataset_dicts = []
    idx = 0
    for cat_id, name in tqdm(enumerate(categories)):
        for i, v in enumerate(paths[name]['originals']):
            idx +=1
            if mode == 'train':
                if idx%10 == 0:
                    continue
            elif mode == 'val':
                if idx%10 != 0:
                    continue
            record = {}

            height, width = cv2.imread(v).shape[:2]

            record["file_name"] = v
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width

            bimasks = gen_masks(paths[name]['masks'][i])
            objs = []
            for mask in bimasks:
                polygon = binary_mask_to_polygon(mask, tolerance = 0)
                px = [v for i, v in enumerate(polygon[0]) if (i+1)%2==1]
                py = [v for i, v in enumerate(polygon[0]) if (i+1)%2==0]
                poly = [(x + 0.5, y) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]


                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    #"category_id": cat_id,
                    "category_id": 0,
                    "iscrowd": 0
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts

for d in ["train", "val"]:
    #DatasetCatalog.register("wildlife_" + d, lambda d=d: get_data_dicts(d))
    #MetadataCatalog.get("wildlife_" + d).set(thing_classes=categories)
    DatasetCatalog.register("wildlifebinary_" + d, lambda d=d: get_data_dicts(d))
    MetadataCatalog.get("wildlifebinary_" + d).set(thing_classes=['subject'])

wildlifebinary_metadata = MetadataCatalog.get("wildlifebinary_train")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("wildlifebinary_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 3000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1#19  # only has one class (ballon)
cfg.OUTPUT_DIR = './output_binary'
try:
    os.rmdir('./output_binary')
    print('rewriting output dir')
except:
    print("creating output dir")
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()






