import os

# files structure
ROOT_DIR = os.path.abspath('')
TEST_IMAGES_DIR = os.path.join(ROOT_DIR, "images/test/")

# Mask RCNN model
MASK_RCNN_MODEL_DIR = os.path.join(ROOT_DIR, 'models/mrcnn/')
MASK_RCNN_LOGS_DIR = os.path.join(MASK_RCNN_MODEL_DIR, "logs/")
MASK_RCNN_LAST_MODEL_WEIGHTS_PATH = os.path.join(MASK_RCNN_MODEL_DIR, "mask_rcnn_credit_card_last.h5")

# external API
VISION_API_URL = 'https://vision.googleapis.com/v1/images:annotate'




