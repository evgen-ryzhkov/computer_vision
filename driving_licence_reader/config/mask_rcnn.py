import os
from models.mrcnn.config import Config

ROOT_DIR = os.path.abspath('')
MASK_RCNN_MODEL_DIR = os.path.join(ROOT_DIR, 'models/mrcnn/')
MASK_RCNN_LOGS_DIR = os.path.join(MASK_RCNN_MODEL_DIR, "logs/")
MASK_RCNN_LAST_MODEL_WEIGHTS_PATH = os.path.join(MASK_RCNN_MODEL_DIR, "mask_rcnn_driving_licence.h5")


class MaskRCNNConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "driving_licence"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9