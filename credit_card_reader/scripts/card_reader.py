"""
Credit card reader
Reading card number and expiry date in weak controlled space (card form, color, position)

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage:

    # read card
    python -m scripts.card_reader --image=<image file name in images/test directory>

"""
import config.access as access_config
import config.main as main_config

import sys
import argparse
import time
import numpy as np

import cv2
import imutils
from imutils.object_detection import non_max_suppression

import base64
import requests
import re


sys.path.append(main_config.ROOT_DIR)  # To find local version of the library
# Import Mask RCNN
from models.mrcnn.config import Config
from models.mrcnn import model as modellib, utils



############################################################
#  Configurations
############################################################


class MaskRCNNConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "credit_card"

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


class CreditCardReader():

    def read_card(self):
        print('[INFO] Loading input image...')
        input_image = self._get_input_image()

        print('[INFO] Loading Mark RCNN model...')
        mask_rcnn_model = self._get_mask_rcnn_model()

        print('[INFO] Getting card instance image...')
        card_image = self._get_object_instance(mask_rcnn_model, input_image)

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

    @staticmethod
    def _get_object_instance(model, image):
        # Detecting objects
        r = model.detect([image], verbose=1)[0]
        found_objects_count = r['class_ids'].shape[-1]

        if found_objects_count > 0:
            # todo There is not processing for case when there are several cards on the image
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
            credit_card_instance = masked_img[roi_box[0]:roi_box[2], roi_box[1]:roi_box[3]]
            return credit_card_instance

        else:
            print('Credit cards were not found.')

    def _prepare_card_for_text_reading(self, card_instance):
        # getting bird eye view of credit card for better text reading
        instance_with_background_around = self._add_background_around_instance_roi(card_instance)
        bird_eye_view_instance = self._get_birds_eye_view_roi(instance_with_background_around)
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

    def _get_birds_eye_view_roi(self, instance_image):
        # algorithm:
        #   0. resize image to smaller size for better perfomance
        #   1. find the biggest contour
        #   2. find 4 vertices
        #   3. perspective transform image by 4 vertices

        # for increase work speed
        # maybe it will need to turn on
        ratio = instance_image.shape[0] / 300.0
        resized_image = imutils.resize(instance_image, height=960)

        biggest_contour = self._get_biggest_contour(resized_image)
        vertices = self._get_vertices(biggest_contour)
        birds_eye_view_image = self._get_birds_eye_view_image(resized_image, vertices)
        return birds_eye_view_image

    def _get_biggest_contour(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # remove high frequency noises
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        lower, upper = self._get_canny_parameters(gray, sigma=0.5)
        edged = cv2.Canny(gray, lower, upper)

        # closed operation in order to contours was closed
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        # find the biggest contours in the edged image
        card_contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest_contour = max(card_contours, key=cv2.contourArea)
        return biggest_contour

    @staticmethod
    def _get_canny_parameters(img_gray, sigma=0.5):
        # automatic choose parameters for Canny detection
        v = np.median(img_gray)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        return lower, upper

    @staticmethod
    def _get_vertices(contour):
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        return box

    @staticmethod
    def _get_birds_eye_view_image(image, four_points):
        # for this prototype it required that card will be in album view
        # there is no processing for portrait view

        # define order of corners
        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        s = four_points.sum(axis=1)
        top_left = four_points[np.argmin(s)]
        bottom_right = four_points[np.argmax(s)]

        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(four_points, axis=1)
        top_right = four_points[np.argmin(diff)]
        bottom_left = four_points[np.argmax(diff)]

        # look for debugging
        # point colors - BGR format
        # cv2.circle(image, (top_left[0], top_left[1]), 5, (255, 0, 0), -1)           # blue
        # cv2.circle(image, (bottom_right[0], bottom_right[1]), 5, (0, 255, 0), -1)   # green
        # cv2.circle(image, (top_right[0], top_right[1]), 5, (0, 0, 255), -1)         # red
        # cv2.circle(image, (bottom_left[0], bottom_left[1]), 5, (0, 255, 255), -1)   # yellow
        # cv2.imshow("The vertices", image)
        # cv2.waitKey(0)

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        width_a = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
        width_b = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
        max_width = max(int(width_a), int(width_b))

        # compute the height of the new image, which will be the
        # maximum distance between the top-right and bottom-right
        # y-coordinates or the top-left and bottom-left y-coordinates
        height_a = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
        height_b = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))
        max_height = max(int(height_a), int(height_b))

        input_vertices = np.float32([top_left, top_right, bottom_right, bottom_left])
        output_vertices = np.array([
            [0, 0],
            [max_width-1, 0],
            [max_width-1, max_height-1],
            [0, max_height-1]
        ], dtype="float32")

        transform_matrix = cv2.getPerspectiveTransform(input_vertices, output_vertices)
        birds_eye_view_image = cv2.warpPerspective(image, transform_matrix, (max_width, max_height))

        return birds_eye_view_image


    def _get_card_number_and_valid_date(self, image):
        image_for_east, east_image_width, east_image_height, ratio_h, ratio_w = self._prepare_image_for_east_detector(image)
        all_text_boxes_on_card = self._get_text_boxes(image_for_east, east_image_width, east_image_height, ratio_h, ratio_w)

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
        card_number = self._read_text_from_roi(card_number_joint_roi)
        if card_number != '':
            # sometimes result of text reading by Google vision isn't look nice
            formatted_card_number = self._format_card_number(card_number)
        else:
            formatted_card_number = 'Card number has not been read.'

        expiry_date = self._get_expiry_date(valid_date_roi_arr)
        if expiry_date == '':
            expiry_date = 'Expiry date has not been read.'

        return formatted_card_number, expiry_date

    @staticmethod
    def _prepare_image_for_east_detector(image):
        # getting card number roi

        # The EAST model requires that your input image dimensions be multiples of 32
        # just choose our image size = 640x640
        east_image_height = 640
        east_image_width = 640
        image_for_east = np.zeros((east_image_height, east_image_width, 3), np.uint8)

        # for good text reading, it need to resize image to size east model required with keeping aspect ration
        (input_image_height, input_image_width) = image.shape[:2]
        input_image_ratio = input_image_width / input_image_height
        resized_image_width = east_image_width
        resized_image_height = int(input_image_height / input_image_ratio)
        resized_image = cv2.resize(image, (resized_image_width, resized_image_height), interpolation=cv2.INTER_AREA)

        ratio_h = input_image_height / float(resized_image_height)
        ratio_w = input_image_width / float(resized_image_width)

        # insert resized_image to image for east model
        image_for_east[0:resized_image_height, 0:resized_image_width] = resized_image

        return image_for_east, east_image_width, east_image_height, ratio_h, ratio_w

    @staticmethod
    def _decode_east_text_predictions(scores, geometry):
        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
        # confidence scores
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the
            # geometrical data used to derive potential bounding box
            # coordinates that surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            # loop over the number of columns
            for x in range(0, numCols):
                # if our score does not have sufficient probability,
                # ignore it
                if scoresData[x] < main_config.EAST_MIN_CONFIDENCE:
                    continue

                # compute the offset factor as our resulting feature
                # maps will be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                # extract the rotation angle for the prediction and
                # then compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # use the geometry volume to derive the width and height
                # of the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                # compute both the starting and ending (x, y)-coordinates
                # for the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                # add the bounding box coordinates and probability score
                # to our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

        # return a tuple of the bounding boxes and associated confidences
        return (rects, confidences)

    def _get_text_boxes(self, image_for_east, east_image_width, east_image_height, ratio_h, ratio_w):

        # define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        output_layers = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

        # load the pre-trained EAST text detector
        print("-- [INFO] loading EAST text detector...")
        net = cv2.dnn.readNet(main_config.EAST_MODEL_PATH)

        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(image_for_east, 1.0, (east_image_width, east_image_height),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)
        start = time.time()
        net.setInput(blob)
        (scores, geometry) = net.forward(output_layers)
        end = time.time()

        # show timing information on text prediction
        print("-- [INFO] text detection took {:.6f} seconds".format(end - start))

        # decode the predictions, then  apply non-maxima suppression to
        # suppress weak, overlapping bounding boxes
        (rects, confidences) = self._decode_east_text_predictions(scores, geometry)
        boxes = non_max_suppression(np.array(rects), probs=confidences)

        text_roi_arr = []
        # because boundary has not very good accuracy
        # adding padding for better text capturing
        roi_padding_x = 5
        roi_padding_y = 20 # sometimes google vision works better if there is some space after roi

        for box in boxes:
            top_left_x = int(box[0] * ratio_w)
            top_left_y = int(box[1] * ratio_h)
            bottom_right_x = int(box[2] * ratio_w)
            bottom_right_y = int(box[3] * ratio_h)

            top_left_x = top_left_x - roi_padding_x
            top_left_y = top_left_y
            bottom_right_x = bottom_right_x + roi_padding_x
            bottom_right_y = bottom_right_y + roi_padding_y

            text_roi_arr.append((top_left_x, top_left_y, bottom_right_x, bottom_right_y))

        return text_roi_arr

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


    def _read_text_from_roi(self, image):
        request_params = {'key': access_config.GOOGLE_VISION_API_KEY}
        body = self._make_request(image)
        response = requests.post(url=main_config.VISION_API_URL, params=request_params, json=body)
        response_json = response.json()

        try:
            text = response_json['responses'][0]['textAnnotations'][0]['description']
        except:
            text = ''
        return text

    def _make_request(self, image):
        image_base_64 = self._convert_img_to_base64(image)
        # languageHints needs because sometimes google vision tryied read numbers as not English characters (Cyrillic for instance)
        return {
          "requests": [
            {
              "features": [
                {
                   "maxResults": 50,
                   "type": "DOCUMENT_TEXT_DETECTION"
                }
              ],
              "image": {
                 'content': image_base_64
              },
              "imageContext": {
                  "languageHints": ["en"],
              }
            }
          ]
        }

    @staticmethod
    def _convert_img_to_base64(image):
        buffer = cv2.imencode('.jpg', image)[1]
        img_base64 = base64.b64encode(buffer)
        img_base64 = img_base64.decode("utf-8")
        return img_base64

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
        canditate_texts =  self._read_text_from_roi(joint_roi)
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
