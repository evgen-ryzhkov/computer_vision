"""
Credit card reader
Reading card number and expiry date in weak controlled space (card form, color, position).

Written by Evgeniy Ryzhkov
Work based on work "Splash of Color" by Mattersport, Inc.

------------------------------------------------------------

Usage:

    # read card
    python -m scripts.card_reader --image=<image file name in images/test directory>

"""

import config.access as access_config
import config.main as main_config

import argparse
import numpy as np
import cv2
import imutils
import re

# Nets
from scripts.nets.mask_rcnn import MaskRCNN
from scripts.nets.east import EastTextDetection
from scripts.nets.google_vision import GoogleVision

# Some addition utilities
from scripts.utils.image_processing import ImageProcessing


class CreditCardReader():

    def __init__(self):
        self.image_pros = ImageProcessing()
        self.google_vision = GoogleVision()

    def read_card(self):
        print('[INFO] Loading input image...')
        input_image = self._get_input_image()

        print('[INFO] Getting card instance ...')
        mask_rcnn = MaskRCNN()
        card_image = mask_rcnn.get_object_instance(image=input_image)
        if card_image == '':
            print('[INFO] Credit cards were not found.')
            exit()

        print('[INFO] Preparing card instance image for text reading...')
        prepared_card_image = self._prepare_card_for_text_reading(card_image)

        print('[INFO] Card reading...')
        card_number, expiry_date = self._get_card_number_and_valid_date(prepared_card_image)

        print('\n[OK] Card reading has been finished successfully:')
        print('-- Card number = {}\n-- Expiry date = {}'.format(card_number, expiry_date))
        self._visualize_result(input_image, card_number, expiry_date)

    @staticmethod
    def _get_command_line_arguments():
        ap = argparse.ArgumentParser()
        ap.add_argument("--image", required=True,
                        help="File name from images/test/")
        args = vars(ap.parse_args())
        return args

    def _get_input_image(self):
        user_arguments = self._get_command_line_arguments()
        img_file = user_arguments['image']
        try:
            input_image = cv2.imread(main_config.TEST_IMAGES_DIR + img_file)
            return input_image
        except:
            print('[ERROR] Image file not found or can not be read.')
            exit()

    @staticmethod
    def _visualize_result(input_image, card_number, expiry_date):
        resized_image = imutils.resize(input_image, height=700)

        font = cv2.FONT_HERSHEY_DUPLEX
        bottom_left_corner_of_text = (10, 30)
        font_scale = 0.6
        font_color = (0, 0, 0)
        line_type = 1

        cv2.putText(resized_image, 'Card number - ' + card_number,
                    bottom_left_corner_of_text,
                    font,
                    font_scale,
                    font_color,
                    line_type)
        bottom_left_corner_of_text = (10, 60)
        cv2.putText(resized_image, 'Expiry date - ' + expiry_date,
                    bottom_left_corner_of_text,
                    font,
                    font_scale,
                    font_color,
                    line_type)

        cv2.imshow("Result", resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _prepare_card_for_text_reading(self, card_instance):
        # getting bird eye view of credit card for better text reading
        instance_with_background_around = self._add_background_around_instance_roi(card_instance)
        bird_eye_view_instance = self.image_pros.get_birds_eye_view_roi(instance_image=instance_with_background_around)
        return bird_eye_view_instance

    @staticmethod
    def _add_background_around_instance_roi(instance_img):
        # for good detecting external contour
        # it's required to be empty space around the object
        # so we create a little bit bigger image for instance with filled background around

        # Create black blank image
        instance_img_height = instance_img.shape[0]
        instance_img_width = instance_img.shape[1]

        instance_with_background_around_height = instance_img_height + 40
        instance_with_background_around_width = instance_img_width + 40
        instance_with_background_around = np.zeros((instance_with_background_around_height, instance_with_background_around_width, 3), np.uint8)

        background_color = (0, 0, 0)
        instance_with_background_around[:] = background_color
        instance_with_background_around[20:instance_img_height+20, 20:instance_img_width+20] = instance_img

        return instance_with_background_around

    def _get_card_number_and_valid_date(self, image):
        east_text_detector = EastTextDetection()
        all_text_boxes_on_card = east_text_detector.get_text_boxes(image=image)

        card_number_text_boxes = self._get_card_number_boxes(all_text_boxes_on_card)

        # for request to google api reducing
        # joint numbers as one image
        big_roi_11 = card_number_text_boxes[0][1]
        big_roi_12 = card_number_text_boxes[3][3]
        big_roi_21 = card_number_text_boxes[0][0]
        big_roi_22 = card_number_text_boxes[3][2]
        card_number_joint_roi = image[big_roi_11:big_roi_12, big_roi_21:big_roi_22]

        valid_date_text_boxes = self._get_valid_date_boxes(all_text_boxes_on_card, card_number_text_boxes, image)
        valid_date_roi_arr = []
        for box in valid_date_text_boxes:
            roi = image[box[1]:box[3], box[0]:box[2]]
            valid_date_roi_arr.append(roi)

        print('-- [INFO] reading card number by Google vision...')
        # prepare image for google vision
        card_number_joint_roi_base64 = self.image_pros.convert_img_to_base64(card_number_joint_roi)

        card_number = self.google_vision.read_text_from_image(card_number_joint_roi_base64)
        if card_number != '':
            # sometimes result of text reading by Google vision isn't look nice
            formatted_card_number = self._format_card_number(card_number)
        else:
            formatted_card_number = 'Card number has not been read.'

        expiry_date = self._get_expiry_date(valid_date_roi_arr)
        if expiry_date == '':
            expiry_date = '[INFO] Expiry date has not been read.'

        return formatted_card_number, expiry_date

    @staticmethod
    def _get_card_number_boxes(boxes_list):
        # card number boxes are:
        # -- four boxes
        # -- y_top_left_corner are approximately equal (in range some small delta)
        # -- width and height are approximately equal
        # -- order boxes depends from x_top_left_corners

        card_number_boxes = None
        y_top_left_corner_delta = 20
        wh_delta = 25

        # find 4 boxes with equal top left corners
        for idx, box in enumerate(boxes_list):
            top_left_corner = box[1]
            box_w =  box[2] - box[0]
            box_h = box[3] - box[1]

            card_number_boxes = []
            card_number_boxes.append(box)
            boxes_count = 1

            boxes_list_without_current_box = boxes_list.copy()
            del boxes_list_without_current_box[idx]

            for box_2 in boxes_list_without_current_box:
                top_left_corner_2 = box_2[1]
                box_w_2 = box_2[2] - box_2[0]
                box_h_2 = box_2[3] - box_2[1]

                if abs(top_left_corner - top_left_corner_2) <= y_top_left_corner_delta and \
                   abs(box_w - box_w_2) <= wh_delta and \
                   abs(box_h - box_h_2) <= wh_delta:
                    boxes_count += 1
                    card_number_boxes.append(box_2)

                if boxes_count == 4:
                    break

            if boxes_count == 4:
                break

        # sort card number boxes in order from left to right by x coordinate on top left corner
        # for easier operation it needs to transform array to numpy array
        sorted_card_number_boxes = np.array(card_number_boxes)
        col = 0
        sorted_card_number_boxes = sorted_card_number_boxes[np.argsort(sorted_card_number_boxes[:, col])]

        return sorted_card_number_boxes

    @staticmethod
    def _get_valid_date_boxes(boxes_list, card_number_text_boxes, image):
        # expiry date box is:
        # -- not card number box (located below)
        # -- side ratio in range 2.2 - 3.6 (was chosen experimentally)
        # -- contains symbol '/' - it will check OCR
        # -- if two blocks on the same line - choose the righter one

        # find bottom line of card_numbers
        # it's max of bottom right corner
        y_bottom_right_corners = card_number_text_boxes[:, [3]]
        card_number_bottom_line = np.amax(y_bottom_right_corners)

        valid_date_candidate_boxes = []
        for box in boxes_list:
            box_y_top_coordinate = box[1]

            if box_y_top_coordinate > card_number_bottom_line:
                box_w = box[2] - box[0]
                box_h = box[3] - box[1]
                box_aspect_ratio = box_w / box_h
                if box_aspect_ratio >= 2.2 and \
                   box_aspect_ratio < 3.6:
                    valid_date_candidate_boxes.append(box)

        return valid_date_candidate_boxes

    @staticmethod
    def _format_card_number(card_number):
        # remove spaces and some noise characters for good formatting
        noise = '[ ,_.]'
        card_number_without_spaces = re.sub(noise, '', card_number)
        formated_card_number = card_number_without_spaces[:4] + ' ' + card_number_without_spaces[4:8] + ' ' + \
                               card_number_without_spaces[8:12] + ' ' + card_number_without_spaces[12:16]

        return formated_card_number

    def _get_expiry_date(self, roi_arr):
        # for google api requests reducing
        # joint candidates roi into one roi
        joint_roi = self._joint_expiry_date_roi(roi_arr)
        joint_roi_base64 = self.image_pros.convert_img_to_base64(joint_roi)
        canditate_texts =  self.google_vision.read_text_from_image(joint_roi_base64)
        expiry_date = self._find_expiry_date_among_candidates(canditate_texts)
        return expiry_date

    @staticmethod
    def _joint_expiry_date_roi(roi_arr):
        # joint_roi_cols = max width among roi + borders_paddings*2 (both side)
        # joint_roi_rows = sum height of all roi + borders_paddings (top side) + joint_roi_space_between_rows * roi_cols
        joint_roi_rows = 0
        joint_roi_cols = 0
        joint_roi_space_between_rows = 20
        joint_roi_space_borders_paddings = 20

        for roi in roi_arr:
            roi_w = roi.shape[1]
            roi_h = roi.shape[0]
            if roi_w > joint_roi_cols:
                joint_roi_cols = roi_w
            joint_roi_rows = joint_roi_rows + roi_h + joint_roi_space_between_rows

        joint_roi_rows += joint_roi_space_borders_paddings
        joint_roi_cols = joint_roi_cols + 2*joint_roi_space_borders_paddings

        joint_roi = np.zeros((joint_roi_rows, joint_roi_cols, 3), np.uint8)

        insert_row_start = joint_roi_space_borders_paddings
        insert_col_start = joint_roi_space_borders_paddings
        for idx, roi in enumerate(roi_arr):
            roi_w = roi.shape[1]
            roi_h = roi.shape[0]
            insert_col_end = insert_col_start + roi_w
            insert_row_end = insert_row_start + roi_h

            joint_roi[insert_row_start:insert_row_end, insert_col_start:insert_col_end] = roi
            insert_row_start = insert_row_end + joint_roi_space_between_rows

        return joint_roi

    @staticmethod
    def _find_expiry_date_among_candidates(canditate_texts):
        # convert texts to array by '\n' for easier further work
        canditate_arr = canditate_texts.split('\n')

        # expiry date is:
        # -- contains /
        # todo there isn't any processing for case when there are several dates on the card
        # it could be solve by comparing the dates and choosing the biggest one
        expiry_date = ''
        for candidate in canditate_arr:
            if '/' in candidate:
                expiry_date = candidate
                break
        return expiry_date


# ----------------------------------------------------
ccr = CreditCardReader()
ccr.read_card()
