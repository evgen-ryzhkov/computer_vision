"""
Driving license reader

Input:
    - Photo of driving license (it could be holding by hand)
Output:
    - Scan of the document: must to have rather pretty view (approximately like it usually have scanned document view)
    - Text data: First, Last Names (Latin letters), Birth date, driving license ID number.

Written by Evgeniy Ryzhkov

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
from operator import itemgetter
from skimage.restoration import estimate_sigma

# Nets
from scripts.nets.mask_rcnn import MaskRCNN
import pytesseract

# Some addition utilities
from scripts.utils.image_processing import ImageProcessing


class DrivingLicenceReader:

    def __init__(self):
        self.image_pros = ImageProcessing()

    def read_card(self):
        print('[INFO] Loading input image...')
        input_img = self._get_input_image()

        print('[INFO] Getting scanned view of driving license...')
        scanned_view_img = self._get_driving_licence_scanned_view(input_img)

        print('[INFO] Reading driver data...')
        last_name, first_name, birth_date, id_driving_licence = self._get_driver_data(scanned_view_img)
        print('[OK] Driving licence has been read. Driver data:')
        print('Name = {} {}\nBirth date = {}\nID driving licence = {}'.format(last_name, first_name, birth_date, id_driving_licence))

        # scale scanned_view for demonstration
        scanned_demo = scanned_view_img.copy()
        scanned_demo = imutils.resize(scanned_demo, height=700)

        cv2.imshow("Scanned_view_img", scanned_demo)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
            return cv2.imread(main_config.TEST_IMAGES_DIR + img_file)
        except FileNotFoundError:
            print('[ERROR] Image file has not been found.')
            raise
        except IOError:
            print('[ERROR] Image file can not been read.')
            raise
        except Exception as e:
            print('[ERROR] Unhandled exception. ', e)
            raise

    # ----- getting object --------------
    def _get_driving_licence_scanned_view(self, brg_img):
        print('-- [INFO] Processing image for object detection...')
        processed_img_for_obj_detection = self._get_processed_image_for_object_detection(brg_img)

        print('-- [INFO] Getting object instance and ROI box...')
        instance_image, roi_box = self._get_object(processed_img_for_obj_detection)

        if len(instance_image) == 0:
            print('[INFO] Driving licence has not been found.')
            exit()
        else:
            scanned_view_imaged = self._get_scanned_view_image(instance_image, roi_box, brg_img, processed_img_for_obj_detection)
            return scanned_view_imaged

    @staticmethod
    def _get_processed_image_for_object_detection(brg_img):
        # decrease time for object detection
        processed_img_h = 960
        processed_img = imutils.resize(brg_img, height=processed_img_h)
        return processed_img

    @staticmethod
    def _get_object(processed_brg_img):
        mask_rcnn = MaskRCNN()
        instance_image, roi_box = mask_rcnn.get_object_instance(image=processed_brg_img)
        return instance_image, roi_box

    @staticmethod
    def _get_original_roi_box(original_img, small_roi_box, scale_size):
        original_roi_box = (small_roi_box * scale_size).astype(int)
        original_roi_img = original_img[original_roi_box[0]:original_roi_box[2], original_roi_box[1]:original_roi_box[3]]
        return original_roi_img

    @staticmethod
    def _get_scale_size(original_img, processed_img):
        original_img_h = original_img.shape[0]
        processed_img_h = processed_img.shape[0]
        return original_img_h / processed_img_h

    def _get_scanned_view_image(self, instance_image, roi_box, original_img, processed_img):
        # scanned img get from input img because for OCR needs high resolution image
        scale_size = self._get_scale_size(original_img, processed_img)
        original_roi_img = self._get_original_roi_box(original_img, roi_box, scale_size)

        # for getting perspective transform_matrix, use instance_image because it's easier
        # to define vertices of the document
        # then, apply transform_matrix to rotate roi_box to get "scanned view"
        scanned_view = self.image_pros.get_birds_eye_view_roi(instance_image=instance_image, original_roi_box=original_roi_img, scale_size=scale_size)
        return scanned_view

    # ----- /getting object --------------

    # ----- reading driver data --------------
    def _get_driver_data(self, brg_scanned_img):
        scaled_img_for_ocr = self._get_scale_img_for_ocr(brg_scanned_img)
        useful_text_roi_arr = self._get_useful_text_roi(scaled_img_for_ocr)

        print('-- [INFO] Reading driver data..')
        last_name, first_name, birth_date, id_driving_licence = self._read_driver_data(useful_text_roi_arr, scaled_img_for_ocr)
        return last_name, first_name, birth_date, id_driving_licence

    @staticmethod
    def _get_scale_img_for_ocr(brg_img):
        # decrease time for object detection
        processed_img_h = 2400
        processed_img = imutils.resize(brg_img, height=processed_img_h)
        return processed_img

    def _get_useful_text_roi(self, brg_img):
        processed_img = self._get_processed_img_for_getting_useful_text_roi(brg_img)

        # getting contours
        all_contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        useful_text_roi = []

        # filtering found contours by size
        for c in all_contours:
            (x, y, w, h) = cv2.boundingRect(c)
            delta = 5
            x -= delta
            y -= delta
            w = w + 2 * delta
            h = h + 2 * delta
            # hard code! for image height = 960px
            if w > (20 + 2 * delta) and h > (20 + 2 * delta) and h < 70:
                useful_text_roi.append((x, y, w, h))

        scaled_useful_text_roi = self._scale_roi_to_input_img(useful_text_roi, brg_img, processed_img)

        return scaled_useful_text_roi

    @staticmethod
    def _get_processed_img_for_getting_useful_text_roi(input_brg_img):

        # resize for better perfomance
        img_for_ocr_h = 960
        resized_img = imutils.resize(input_brg_img, height=img_for_ocr_h)

        # some processing
        gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

        # remove some noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # -- lightning dark pieces (text is black)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7))  # for 960
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, thresh = cv2.threshold(blackhat, 0.0, 255.0, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # remove some noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # -- for better searching of rectangular elements
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
        connected_tresh = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)

        return connected_tresh

    @staticmethod
    def _scale_roi_to_input_img(useful_text_roi, input_brg_img, processed_img):
        h_input_img = input_brg_img.shape[0]
        h_processed_img = processed_img.shape[0]
        scale_size = h_input_img / h_processed_img

        np_arr = np.array(useful_text_roi)
        scaled_text_roi_arr = np_arr * scale_size
        scaled_text_roi_arr = scaled_text_roi_arr.astype(int)
        return scaled_text_roi_arr

    def _read_driver_data(self, text_roi_arr, brg_img):

        processed_img_for_ocr = self._get_processed_image_for_ocr(brg_img)

        # useful data are located against row numbers and nearby them
        # so find document row numbers and save their coordinates
        row_1_top, row_2_top, row_3_top, row_5_top, numbers_vertical_line \
            = self._get_row_coordinates(text_roi_arr, processed_img_for_ocr)

        # finding of necessary text boxes among all roi text boxes
        driver_data_boxes = self._get_driver_data_boxes(text_roi_arr, row_1_top, row_2_top, row_3_top, row_5_top,
                                                        numbers_vertical_line)

        # reading text from driver data image boxes by tesseract
        last_name, first_name, birth_date, id_driving_licence = self._ocr_driver_data(processed_img_for_ocr, driver_data_boxes)

        # there may be postprocess

        return last_name, first_name, birth_date, id_driving_licence

    @staticmethod
    def _get_processed_image_for_ocr(brg_img):

        gray = cv2.cvtColor(brg_img, cv2.COLOR_BGR2GRAY)

        # remove some noise
        # define noise level
        sigma_est = estimate_sigma(gray, multichannel=False, average_sigmas=True)

        # for different noise level, use different denoising approach
        if sigma_est > 3 and sigma_est < 7:
            gray = cv2.medianBlur(gray, 5)
        elif sigma_est >= 7:
            gray = cv2.medianBlur(gray, 7)
        else:
            gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # -- lightning dark pieces (text is black)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (39, 39))  # for 960
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        _, thresh = cv2.threshold(blackhat, 0.0, 255.0, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.dilate(thresh, kernel, iterations=1)

        return erosion

    @staticmethod
    def _get_row_coordinates(text_roi_arr, processed_img):

        # first find roi are similar row numbers
        row_numbers = []

        for text in text_roi_arr:
            aspect_ratio = text[2] / text[3]
            if aspect_ratio > 0.9 and aspect_ratio < 2:
                row_numbers.append(text)


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

            # sometimes tesseract read '.' as ','
            text_number = re.sub(r",", ".", text_number)

            if text_number == '1.':
                row_1_top = number[1]
                numbers_vertical_line = number[0]
            if text_number == '2.':
                row_2_top = number[1]
                # if '1.' has not been recognized
                # define number vertical lines by '2.'
                numbers_vertical_line = number[0]
            if text_number == '3.':
                row_3_top = number[1]
            if text_number == '5.':
                row_5_top = number[1]

        return row_1_top, row_2_top, row_3_top, row_5_top, numbers_vertical_line

    @staticmethod
    def _get_driver_data_boxes(text_roi_arr, row_1_top, row_2_top, row_3_top, row_5_top, numbers_vertical_line):

        first_name_applicants = []
        latin_last_name_box, birth_date_box, id_driving_licence_box = np.zeros(3, dtype=int), np.zeros(3,
                                                                                                       dtype=int), np.zeros(
            3, dtype=int)
        delta_row = 40  # take into account some error of text box positions
        delta_v_line_low = 40
        delta_v_line_high = 250  # todo - hard code, it should change on dependence value from scale factor (100 * scale)

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
        if len(first_name_applicants) == 0:
            latin_first_name_box = np.zeros(3, dtype=int)
        else:
            min_y_coordinate = min(x[1] for x in first_name_applicants)
            first_name_applicants = [x for x in first_name_applicants if x[1] > min_y_coordinate]
            first_name_applicants = sorted(first_name_applicants, key=itemgetter(1))
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
            if not np.any(box):
                user_data.append('Not defined')
            else:
                x0 = box[1]
                x1 = box[1] + box[3]
                y0 = box[0]
                y1 = box[0] + box[2]
                img_roi = processed_img[x0:x1, y0:y1]

                user_data.append(pytesseract.image_to_string(img_roi, config='--psm 8'))

        last_name = user_data[0]
        first_name = user_data[1]
        birth_date = user_data[2]
        id_driving_licence = user_data[3]
        return last_name, first_name, birth_date, id_driving_licence
    # ----- /reading driver data --------------


# ----------------------------------------------------
dlr = DrivingLicenceReader()
dlr.read_card()
