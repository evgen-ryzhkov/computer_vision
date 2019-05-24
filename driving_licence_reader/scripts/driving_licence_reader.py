"""
Credit card reader
Reading card number and expiry date in weak controlled space (card form, color, position).

Written by Evgeniy Ryzhkov
Work based on work "Splash of Color" by Mattersport, Inc.

------------------------------------------------------------

Usage:

    # read card
    python -m scripts.driving_licence_reader --image=<image file name in images/test directory>

"""

import config.main as main_config

import argparse
import numpy as np
import cv2
import imutils
import re
import pytesseract
from operator import itemgetter

# Nets
from scripts.nets.mask_rcnn import MaskRCNN

# Some addition utilities
from scripts.utils.image_processing import ImageProcessing

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import hsv_to_rgb

import time

class DrivingLicenceReader():

    def __init__(self):
        self.image_pros = ImageProcessing()

    def read_card(self):
        print('[INFO] Loading input image...')
        img_for_ocr, img_for_processing, scale_size = self._get_input_image()

        # print('[INFO] Getting driving licence instance ...')
        # mask_rcnn = MaskRCNN()
        # instance_image, roi_box = mask_rcnn.get_object_instance(image=input_image)
        # if len(instance_image) == 0:
        #     print('[INFO] Driving licence has not been found.')
        #     exit()

        # cv2.imshow("Roi box", roi_box)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # todo going to try to regulate card position by the interface
        # print('[INFO] Card instance image processing for text reading...')
        # processed_roi_box = self._prepare_card_for_text_reading(input_image, instance_image, roi_box)
        #
        # cv2.imshow("Result", prepared_card_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        print('[INFO] Reading driver data ...')
        # processed_roi_box, roi_box = input_image, input_image # todo remove after debugging
        useful_text_rois = self._get_useful_text_roi(brg_img=img_for_processing)

        # scale text roi coordinates to img_for_ocr
        np_arr = np.array(useful_text_rois)
        scaled_text_roi_arr = np_arr * scale_size
        scaled_text_roi_arr = scaled_text_roi_arr.astype(int)


        # last_name, first_name, birth_date, id_driving_licence = self._get_driver_data(useful_text_rois, img_for_processing)
        last_name, first_name, birth_date, id_driving_licence = self._get_driver_data(scaled_text_roi_arr, img_for_ocr)
        # last_name, first_name, birth_date, id_driving_licence = self._get_driver_data_alt(img_for_ocr, useful_text_rois, scale_size)

        print('[OK] Driving licence has been read. Driver data:')
        print('Name = {} {}\nBirth date = {}\nID driving licence = {}'.format(last_name, first_name, birth_date, id_driving_licence))

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
        # img_height = 1600
        try:
            input_image = cv2.imread(main_config.TEST_IMAGES_DIR + img_file)

            # debugging
            # img_h, img_w, _ = input_image.shape
            # scale_size = img_height / img_h
            # print('scale_size =', scale_size)
            # new_img_w = int(img_w * scale_size)
            # new_img_h = int(img_h * scale_size)
            # resized_image = cv2.resize(input_image, (new_img_w, new_img_h), interpolation=cv2.INTER_AREA)
            # cv2.imshow("Result", resized_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # debugging
            img_for_ocr_h = 1600
            img_for_processing_h = 960
            scale_size = img_for_ocr_h / img_for_processing_h

            img_for_ocr = imutils.resize(input_image, height=1600)
            img_for_processing = imutils.resize(input_image, height=960)

            # resized_image = input_image
            return img_for_ocr, img_for_processing, scale_size
        except:
            print('[ERROR] Image file not found or can not be read.')
            exit()

    @staticmethod
    def _get_useful_text_roi(brg_img):

        start = time.time()
        # some processing
        gray = cv2.cvtColor(brg_img, cv2.COLOR_BGR2GRAY)

        # -- lightning dark pieces (text is black)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)) for 960
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        _, thresh = cv2.threshold(blackhat, 0.0, 255.0, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # -- for better searching of rectangular elements
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1))
        connected_tresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # getting contours
        all_contours, _ = cv2.findContours(connected_tresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        useful_text_roi = []

        # filtering found contours by size
        for c in all_contours:
            (x, y, w, h) = cv2.boundingRect(c)
            delta = 5
            x -= delta
            y -= delta
            w = w + 2*delta
            h = h + 2*delta
            if w > (20 + 2*delta) and h > (20 + 2*delta) and h < 70:
                useful_text_roi.append((x, y, w, h))

        end = time.time()
        print('Getting useful roi took {} sec'.format(end - start))
        # 0.0029 for img_h = 640
        # 0.00695 for img_h = 960
        # 0.159 for img_h = 1600
        return useful_text_roi

    def _get_driver_data_alt(self, img_for_ocr, text_roi_arr, scale_size):
        img_c = img_for_ocr.copy()

        # scale text roi coordinates to img_for_ocr
        np_arr = np.array(text_roi_arr)
        scaled_text_roi_arr = np_arr * scale_size
        scaled_text_roi_arr = scaled_text_roi_arr.astype(int)

        row_1_top, row_2_top, row_3_top, row_5_top, numbers_vertical_line \
            = self._get_row_coordinates(scaled_text_roi_arr, img_for_ocr)

        driver_data_boxes = self._get_driver_data_boxes(scaled_text_roi_arr, row_1_top, row_2_top, row_3_top, row_5_top,
                                                        numbers_vertical_line)
        last_name, first_name, birth_date, id_driving_licence = self._ocr_driver_data(img_for_ocr, driver_data_boxes)

        # for text in scaled_text_roi_arr:
        #     cv2.rectangle(img_c, (text[0], text[1]), (text[0] + text[2] - 1, text[1] + text[3] - 1), (0, 255, 0), 1)
        # cv2.imshow("Useful text rois", img_c)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


        return '1', '1', '1', '1'


    def _get_driver_data(self, text_roi_arr, brg_img):

        # for better OCR do some processing
        processed_img = self._get_processed_image_for_text_reading(brg_img)

        print('Processed img shape = ', processed_img.shape)
        # cv2.namedWindow("Processing for OCR", cv2.WINDOW_NORMAL)
        cv2.imshow("Processing for OCR", processed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # useful data are located against row numbers and nearby them
        # so find document row numbers and save their coordinates
        row_1_top, row_2_top, row_3_top, row_5_top, numbers_vertical_line\
            = self._get_row_coordinates(text_roi_arr, processed_img)

        # finding of necessary text boxes among all roi text boxes
        driver_data_boxes = self._get_driver_data_boxes(text_roi_arr, row_1_top, row_2_top, row_3_top, row_5_top, numbers_vertical_line)

        # reading text from driver data image boxes by tesseract
        last_name, first_name, birth_date, id_driving_licence = self._ocr_driver_data(processed_img, driver_data_boxes)

        # cv2.rectangle(closed, (latin_last_name_box[0], latin_last_name_box[1]),
        #               (latin_last_name_box[0] + latin_last_name_box[2] - 1, latin_last_name_box[1] + latin_last_name_box[3] - 1),
        #               (0, 255, 0), 1)
        # cv2.rectangle(closed, (latin_first_name_box[0], latin_first_name_box[1]),
        #               (latin_first_name_box[0] + latin_first_name_box[2] - 1,
        #                latin_first_name_box[1] + latin_first_name_box[3] - 1),
        #               (0, 255, 0), 1)
        # cv2.rectangle(closed, (birth_date[0], birth_date[1]),
        #               (birth_date[0] + birth_date[2] - 1, birth_date[1] + birth_date[3] - 1),
        #               (0, 255, 0), 1)
        # cv2.rectangle(closed, (id_driving_licence[0], id_driving_licence[1]),
        #               (id_driving_licence[0] + id_driving_licence[2] - 1, id_driving_licence[1] + id_driving_licence[3] - 1),
        #               (0, 255, 0), 1)

        # for p in first_name_pret:
        #     cv2.rectangle(img_c_2, (p[0], p[1]), (p[0] + p[2] - 1, p[1] + p[3] - 1), (0, 255, 0), 1)

        # cv2.imshow("Last name + first name + birth date + id licence", img_roi)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return last_name, first_name, birth_date, id_driving_licence

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

    def _prepare_card_for_text_reading(self, input_image, card_instance, roi_box):
        # getting bird eye view of credit card for better text reading
        # instance_with_background_around = self._add_background_around_instance_roi(card_instance)
        instance_with_background_around = card_instance
        bird_eye_view_instance = self.image_pros.get_birds_eye_view_roi(full_image=input_image, image=instance_with_background_around, roi_box=roi_box)
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

    @staticmethod
    def _get_processed_image_for_text_reading(brg_img):

        # brg_img = cv2.imread(main_config.TEST_IMAGES_DIR + 'text_11.jpg')
        gray = cv2.cvtColor(brg_img, cv2.COLOR_BGR2GRAY)
        gray_inv = cv2.bitwise_not(gray)
        cv2.imshow("Gray", gray_inv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
        # tophat = cv2.morphologyEx(gray_inv, cv2.MORPH_TOPHAT, kernel)
        #
        # cv2.imshow("top hat", tophat)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 19))
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        # blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        #
        # cv2.imshow("blackhat", blackhat)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # _, image_spec = cv2.threshold(blackhat, 0.0, 255.0, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # _, image_spec = cv2.threshold(gray, 130.0, 255.0, cv2.THRESH_BINARY)
        _, image_spec = cv2.threshold(gray_inv, 150.0, 255.0, cv2.THRESH_BINARY_INV)
        # image_spec = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        cv2.imshow("Thresh", image_spec)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # remove some noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(image_spec, cv2.MORPH_CLOSE, kernel)

        #
        # cv2.imshow("closed", closed)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # kernel = np.ones((3, 3), np.uint8)
        # erosion = cv2.dilate(closed, kernel, iterations=1)

        # cv2.imshow("erosion", erosion)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return closed

    @staticmethod
    def _get_row_coordinates(text_roi_arr, processed_img):

        # first find roi are similar row numbers
        row_numbers = []
        # image_c_1 = brg_img.copy()

        for text in text_roi_arr:
            aspect_ratio = text[2] / text[3]
            if aspect_ratio > 0.9 and aspect_ratio < 2:
                row_numbers.append(text)
                # cv2.rectangle(image_c_1, (text[0], text[1]), (text[0] + text[2] - 1, text[1] + text[3] - 1), (0, 255, 0), 1)

        row_1_top, row_2_top, row_3_top, row_5_top = 0, 0, 0, 0
        numbers_vertical_line = 0

        for number in row_numbers:
            x0 = number[1]
            x1 = number[1] + number[3]
            y0 = number[0] - 15
            y1 = number[0] + number[2] + 15
            img_number = processed_img[x0:x1, y0:y1]
            text_number = pytesseract.image_to_string(img_number,
                                                      config='-c tessedit_char_whitelist=1234567890 --psm 10')

            # print(text_number)
            # sometimes tesseract read '.' as ','
            text_number = re.sub(r",", ".", text_number)

            if text_number == '1.':
                row_1_top = number[1]
                numbers_vertical_line = number[0]
            if text_number == '2.':
                row_2_top = number[1]
            if text_number == '3.':
                row_3_top = number[1]
            if text_number == '5.':
                row_5_top = number[1]

        if row_1_top == 0 or \
                row_2_top == 0 or \
                row_3_top == 0 or \
                row_5_top == 0:
            print('[INFO] Number rows have not been read.')
            exit()

        return row_1_top, row_2_top, row_3_top, row_5_top, numbers_vertical_line

    @staticmethod
    def _get_driver_data_boxes(text_roi_arr, row_1_top, row_2_top, row_3_top, row_5_top, numbers_vertical_line):
        # img_c_2 = brg_img.copy()

        first_name_applicants = []
        latin_last_name_box, birth_date_box, id_driving_licence_box = (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)
        delta_row = 20  # take into account some error of text box positions
        delta_v_line_low = 10
        delta_v_line_high = 150

        for text in text_roi_arr:
            text_top_corner = text[1]
            text_left_corner = text[0]
            text_left_corner_minus_v_line = text_left_corner - numbers_vertical_line

            # all needed data are not far right (<100px) from numbers vertical line
            if text_left_corner_minus_v_line < delta_v_line_high and \
                    text_left_corner_minus_v_line > delta_v_line_low:

                # finding last name
                # last name lays between row_1_top and row_2_top
                # we need only latin name
                # latin name is lower in the row
                if (text_top_corner + delta_row) > row_1_top and \
                        (text_top_corner + delta_row) < row_2_top and \
                        text_top_corner > latin_last_name_box[1]:
                    latin_last_name_box = text

                # finding first name step 1 - getting applicants from 2nd rows
                # first name lays between row_2_top and row_3_top
                # we need only latin name
                # latin name is always on the 2nd substring of the first name section
                if (text_top_corner + delta_row) > row_2_top and \
                        (text_top_corner + delta_row) < row_3_top:
                    first_name_applicants.append(text)

                # finding birth date
                # it lays in 3rd row and nearby number of the row (3.)
                if text_top_corner > (row_3_top - delta_row) and \
                        text_top_corner < (row_3_top + delta_row):
                    birth_date_box = text

                # finding id driving licence
                # it lays in 5rd row and nearby number of the row (5.)
                if text_top_corner > (row_5_top - delta_row) and \
                        text_top_corner < (row_5_top + delta_row):
                    id_driving_licence_box = text

        # getting first name step 2
        # getting second substrings by second value of y coordinate of top lef corner
        # print('Pret before sort ', first_name_applicants)
        min_y_coordinate = min(x[1] for x in first_name_applicants)
        # print('min_v=', min_v)
        first_name_applicants = [x for x in first_name_applicants if x[1] > min_y_coordinate]
        # print('Pret after delete', first_name_pret)
        first_name_applicants = sorted(first_name_applicants, key=itemgetter(1))
        # print('Pret after sort', first_name_pret)
        latin_first_name_box = first_name_applicants[0]

        driver_data_boxes = []
        driver_data_boxes.append(latin_last_name_box)
        driver_data_boxes.append(latin_first_name_box)
        driver_data_boxes.append(birth_date_box)
        driver_data_boxes.append(id_driving_licence_box)

        return driver_data_boxes

    @staticmethod
    def _ocr_driver_data(processed_img, driver_data_boxes):
        user_data = []
        for box in driver_data_boxes:
            x0 = box[1]
            x1 = box[1] + box[3]
            y0 = box[0]
            y1 = box[0] + box[2]
            img_roi = processed_img[x0:x1, y0:y1]
            # cv2.imshow('Number ', img_roi)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            user_data.append(pytesseract.image_to_string(img_roi, config='--psm 8'))

        last_name = user_data[0]
        first_name = user_data[1]
        birth_date = user_data[2]
        id_driving_licence = user_data[3]
        return last_name, first_name, birth_date, id_driving_licence


# ----------------------------------------------------
dlr = DrivingLicenceReader()
dlr.read_card()
