"""
Finding object and object instance (Cutted part of input image by mask)

"""

import config.main as main_config
import sys
import cv2

sys.path.append(main_config.ROOT_DIR)  # To find local version of the library
from models.mrcnn import model as modellib, utils
from config.mask_rcnn import MaskRCNNConfig


class MaskRCNN:

    def get_object_instance(self, image):
        print('-- [INFO] Loading Mask RCNN model...')
        model = self._get_mask_rcnn_model()

        print('-- [INFO] Detecting objects...')
        r = model.detect([image], verbose=1)[0]
        found_objects_count = r['class_ids'].shape[-1]

        if found_objects_count > 0:
            # todo There is not processing for case when there are several objects on the image
            first_credit_card_instance = {
                'roi': r['rois'][0],
                'scores': r['scores'][0],
                'mask': r['masks'][:, :, 0]
            }
            roi_box = first_credit_card_instance['roi']
            mask = first_credit_card_instance['mask']

            # maybe it could be done easier
            # convert the mask from a boolean to an integer mask with
            # to values: 0 or 255, then apply the mask
            vis_mask = (mask * 255).astype("uint8")
            masked_img = cv2.bitwise_and(image, image, mask=vis_mask)

            # getting masked roi
            object_instance = masked_img[roi_box[0]:roi_box[2], roi_box[1]:roi_box[3]]
        else:
            object_instance = ''

        return object_instance

    @staticmethod
    def _get_mask_rcnn_model():
        mask_rcnn_config = MaskRCNNConfig()
        model = None
        try:
            model = modellib.MaskRCNN(mode="inference", config=mask_rcnn_config,
                                      model_dir=main_config.MASK_RCNN_LOGS_DIR)
        except:
            print('[ERROR] Something wrong with creating model.')
            exit()

        try:
            model.load_weights(main_config.MASK_RCNN_LAST_MODEL_WEIGHTS_PATH, by_name=True)
            return model
        except:
            print('[ERROR] Something wrong with weights for Mask RCNN.')
            exit()
