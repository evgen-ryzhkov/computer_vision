# how to inspect traning data https://github.com/matterport/Mask_RCNN/blob/master/samples/coco/inspect_data.ipynb

import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)
import mrcnn.utils
import mrcnn.visualize as visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

import scripts.object_detector as od

config = od.CreditCardConfig()
INSPECT_IMAGES_DIR = os.path.join(ROOT_DIR, "datasets")

# Load dataset
# Get the dataset from the releases page
# https://github.com/matterport/Mask_RCNN/releases
dataset = od.CreditCardDataset()
dataset.load_credit_card(INSPECT_IMAGES_DIR, "train")

# Must call before using the dataset
dataset.prepare()

# Load and display random samples
image_ids = np.random.choice(dataset.image_ids, 4)
for image_id in image_ids:
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset.class_names)