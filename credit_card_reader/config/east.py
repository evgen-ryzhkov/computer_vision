import config.main as main_config
import os

EAST_MODEL_DIR = os.path.join(main_config.ROOT_DIR, 'models/east_text_detector/')
EAST_MODEL_PATH = os.path.join(EAST_MODEL_DIR, "frozen_custom_east_text_detection.pb")
EAST_MIN_CONFIDENCE = 0.5