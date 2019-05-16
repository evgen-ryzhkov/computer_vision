"""
Finding object and object instance (Cutted part of input image by mask)

"""

import sys
import cv2
import config.main as main_config
import config.mask_rcnn
import numpy as np

sys.path.append(main_config.ROOT_DIR)  # To find local version of the library
from models.mrcnn import model as modellib


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
            # vis_mask = (mask * 255).astype("uint8")
            # masked_img = cv2.bitwise_and(image, image, mask=vis_mask)

            # getting object roi
            # for better capture of the documents, do some increasing of roi_box
            start_delta = 40
            img_h, img_w, _ = image.shape
            delta_roi_box = self._get_optimal_delta_roi_box(roi_box, start_delta, img_h, img_w)
            # object_instance = masked_img[roi_box[0]:roi_box[2], roi_box[1]:roi_box[3]]
            object_instance = image[delta_roi_box[0]:delta_roi_box[2], delta_roi_box[1]:delta_roi_box[3]]
        else:
            object_instance = []

        return object_instance

    def _get_optimal_delta_roi_box(self, roi_box, delta, img_h, img_w):
        delta_roi_box = np.copy(roi_box)
        delta_step = 10
        # если дельта досигла своего минимума, возвращаем исходный результат
        if delta < delta_step:
            return roi_box

        else:
            delta_roi_box[0] = roi_box[0] - delta
            delta_roi_box[1] = roi_box[1] - delta
            delta_roi_box[2] = roi_box[2] + delta
            delta_roi_box[3] = roi_box[3] + delta

            # если вышли за границы изображения, уменьшаем дельту и пробуем снова
            if delta_roi_box[0] < 0 or \
               delta_roi_box[1] < 0 or \
               delta_roi_box[2] > img_h or \
               delta_roi_box[3] > img_w:
                delta -= delta_step
                return self._get_optimal_delta_roi_box(roi_box, delta, img_h, img_w)

            # если за границы изображения не вышли, значит нашли максимальную дельту
            else:
                return delta_roi_box

    @staticmethod
    def _get_mask_rcnn_model():
        mask_rcnn_config = config.mask_rcnn.MaskRCNNConfig()
        model = None
        try:
            model = modellib.MaskRCNN(mode="inference", config=mask_rcnn_config,
                                      model_dir=config.mask_rcnn.MASK_RCNN_LOGS_DIR)
        except:
            print('[ERROR] Something wrong with creating model.')
            exit()

        try:
            model.load_weights(config.mask_rcnn.MASK_RCNN_LAST_MODEL_WEIGHTS_PATH, by_name=True)
            return model
        except:
            print('[ERROR] Something wrong with weights for Mask RCNN.')
            exit()
