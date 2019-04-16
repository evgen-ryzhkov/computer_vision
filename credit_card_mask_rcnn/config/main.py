import os

ROOT_DIR = os.path.abspath('')
LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
MODEL_DIR = os.path.join(ROOT_DIR, 'mrcnn')
COCO_WEIGHTS_PATH = os.path.join(MODEL_DIR, "mask_rcnn_coco.h5")

DOWNLADED_IMG_DIR = os.path.join(ROOT_DIR, 'datasets/downloaded/')
FILTERED_IMG_DIR = os.path.join(ROOT_DIR, 'datasets/filtered/')
TRAIN_IMG_DIR = os.path.join(ROOT_DIR, 'datasets/train/')
VAL_IMG_DIR = os.path.join(ROOT_DIR, 'datasets/val/')

