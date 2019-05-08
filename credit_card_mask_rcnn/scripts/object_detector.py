"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python object_detector.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python object_detector.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python object_detector.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Debugging
    python object_detector.py read_card --image=<image file name in images/test directory>

"""


# import config.main as config

import os
import sys
import json
import datetime
import time
import numpy as np
import skimage.draw

import cv2
import imutils
from imutils.object_detection import non_max_suppression

from google.cloud import vision
import io
import base64
import requests



# # Import Mask RCNN
# sys.path.append(config.ROOT_DIR)  # To find local version of the library
# from mrcnn.config import Config
# from mrcnn import model as modellib, utils

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

import config.access as access_config
import config.main as main_config

# todo - remove settings to main config
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mrcnn/mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Path to test images dir
TEST_IMAGES_DIR = os.path.join(ROOT_DIR, "images/test/")

# Path to last trained weights
LAST_MODEL_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mrcnn/mask_rcnn_credit_card_last.h5")

EAST_MODEL_PATH = os.path.join(ROOT_DIR, "east_text_detector/frozen_custom_east_text_detection.pb")
EAST_MIN_CONFIDENCE = 0.5

############################################################
#  Configurations
############################################################


class CreditCardConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "credit_card"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class CreditCardDataset(utils.Dataset):

    def load_credit_card(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("credit_card", 1, "credit_card")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

                # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            # image = skimage.io.imread(image_path) # error with reading png
            # except:
            #     print('[ERROR] Troubles with file ', a['filename'])
            brg_image = cv2.imread(image_path)
            image = cv2.cvtColor(brg_image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]

            self.add_image(
                "credit_card",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "credit_card":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "credit_card":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CreditCardDataset()
    dataset_train.load_credit_card(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CreditCardDataset()
    dataset_val.load_credit_card(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=50,
                layers='heads')


def read_card(model, img_file=None):
    input_image = cv2.imread(TEST_IMAGES_DIR + img_file)
    card_image = _get_object_instance(model, input_image)

    # cv2.imshow("Card image", card_image)
    # cv2.waitKey(0)

    prepared_card_image = _prepare_card_for_text_reading(card_image)
    # prepared_card_image = input_image

    # cv2.imshow("Debugging", prepared_card_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # todo - remove after debugging
    # prepared_card_image = input_image

    print('[INFO] Card reading...')
    card_number = _get_card_number_and_valid_date(prepared_card_image)
    expiry_date = '??'
    print('\n[OK] Card reading has been finished successfully:')
    print('-- Card number = {}\n'.format(card_number))
    print('-- Expiry date = ', expiry_date)


def _get_object_instance(model, image):
    # Detect objects
    r = model.detect([image], verbose=1)[0]
    found_objects_count = r['class_ids'].shape[-1]

    if found_objects_count > 0:
        # For more simple debugging will work with the first instance only
        first_credit_card_instance = {
            'roi': r['rois'][0],
            'scores': r['scores'][0],
            'mask': r['masks'][:, :, 0]
        }

        # showing box roi
        roi_box = first_credit_card_instance['roi']

        # look at bbox for debugging
        # credit_card_bbox = image[roi_box[0]:roi_box[2], roi_box[1]:roi_box[3]]
        # cv2.imshow("Box roi", credit_card_bbox)
        # cv2.waitKey(0)

        # showing masked roi
        mask = first_credit_card_instance['mask']

        # maybe it could be done easier
        # convert the mask from a boolean to an integer mask with
        # to values: 0 or 255, then apply the mask
        vis_mask = (mask * 255).astype("uint8")
        masked_img = cv2.bitwise_and(image, image, mask=vis_mask)

        # cv2.imshow("Masked img", masked_img)
        # cv2.waitKey(0)

        # getting masked roi
        credit_card_instance = masked_img[roi_box[0]:roi_box[2], roi_box[1]:roi_box[3]]
        return credit_card_instance

    else:
        print('Credit cards were not found.')


def _prepare_card_for_text_reading(card_instance):
    instance_with_background_around = _add_background_around_instance_roi(card_instance)
    bird_eye_view_instance = _get_birds_eye_view_roi(instance_with_background_around)
    return bird_eye_view_instance


def _add_background_around_instance_roi(instance_img):
    # for good detecting external contour
    # it's required to be empty space around the object
    # so we create a little bit bigger image for instance with filled background around

    # Create black blank image
    instance_img_height = instance_img.shape[0]
    instance_img_width = instance_img.shape[1]

    instance_with_background_around_height = instance_img_height + 10
    instance_with_background_around_width = instance_img_width + 10
    instance_with_background_around = np.zeros((instance_with_background_around_height, instance_with_background_around_width, 3), np.uint8)

    # for better detection card contour, background has to be contrast to card color
    # if card light - background dark, else backgound light
    gray = cv2.cvtColor(instance_img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    print('[DEBUGGING] Brightness = ', brightness)
    # brightness threshold was chosen experimentally
    # todo - it needs to debug here
    brightness_threshold = 90
    if brightness < brightness_threshold:
        background_color = (0, 0, 0)
    else:
        background_color = (0, 0, 0)

    instance_with_background_around[:] = background_color
    instance_with_background_around[4:instance_img_height+4, 4:instance_img_width+4] = instance_img

    # cv2.imshow("Instance with background around", instance_with_background_around)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return instance_with_background_around


def _get_birds_eye_view_roi(instance_image):
    # algorithm:
    #   0. resize image to smaller size for better perfomance
    #   1. find the biggest contour
    #   2. find 4 vertices
    #   3. perspective transform image by 4 vertices

    # for increase work speed
    # maybe it will need to turn on
    ratio = instance_image.shape[0] / 300.0
    resized_image = imutils.resize(instance_image, height=960)

    biggest_contour = _get_biggest_contour(resized_image)
    vertices = _get_vertices(biggest_contour)
    birds_eye_view_image = _get_birds_eye_view_image(resized_image, vertices)
    return birds_eye_view_image


def _get_biggest_contour(image):
    # convert the image to grayscale, blur it, and find edges
    # in the image
    # image = cv2.imread(TEST_IMAGES_DIR + 'img_5.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)

    # closed operation in order to contours was closed
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # find the biggest contours in the edged image
    card_contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    biggest_contour = max(card_contours, key=cv2.contourArea)
    return biggest_contour


def _get_vertices(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box


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
    cv2.circle(image, (top_left[0], top_left[1]), 5, (255, 0, 0), -1)           # blue
    cv2.circle(image, (bottom_right[0], bottom_right[1]), 5, (0, 255, 0), -1)   # green
    cv2.circle(image, (top_right[0], top_right[1]), 5, (0, 0, 255), -1)         # red
    cv2.circle(image, (bottom_left[0], bottom_left[1]), 5, (0, 255, 255), -1)   # yellow
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

    # look for debugging
    # cv2.imshow("The vertices", birds_eye_view_image)
    # cv2.waitKey(0)
    return birds_eye_view_image


def _get_card_number_and_valid_date(image):

    image_for_east, east_image_width, east_image_height, ratio_h, ratio_w = _prepare_image_for_east_detector(image)
    # cv2.imshow("Debugging text rois", image_for_east)
    # cv2.waitKey(0)

    all_text_boxes_on_card = _get_text_boxes(image_for_east, east_image_width, east_image_height, ratio_h, ratio_w)

    # debugging
    # border_color = (0, 0, 255)
    # for box in all_text_boxes_on_card:
    #     cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), border_color, 2)
    # cv2.imshow("Debugging text rois", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    card_number_roi_arr = []
    card_number_text_boxes = _get_card_number_boxes(all_text_boxes_on_card)
    # for idx, box in enumerate(card_number_text_boxes):
    #     # cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), border_color, 2)
    #     roi = image[box[1]:box[3], box[0]:box[2]]
    #     card_number_roi_arr.append(roi)

    # for request to google api reducing
    # joint number as one image
    big_roi_11 = card_number_text_boxes[0][1]
    big_roi_12 = card_number_text_boxes[3][3]
    big_roi_21 = card_number_text_boxes[0][0]
    big_roi_22 = card_number_text_boxes[3][2]

    card_number_joint_roi = image[big_roi_11:big_roi_12, big_roi_21:big_roi_22]

    # debugging of digits reading
    # cv2.imwrite(TEST_IMAGES_DIR + 'test_2.jpg', card_number_joint_roi)

    # cv2.imshow("Debugging big roi", card_number_joint_roi)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    valid_date_text_boxes = _get_valid_date_boxes(all_text_boxes_on_card, card_number_text_boxes, image)

    valid_date_roi_arr = []
    for box in valid_date_text_boxes:
        # cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), border_color, 2)
        roi = image[box[1]:box[3], box[0]:box[2]]
        valid_date_roi_arr.append(roi)
    # cv2.imshow("Debugging text rois", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print('[DEBUGGING] card number roi ', len(card_number_roi_arr))
    # for roi in card_number_roi_arr:
    #
    #     number = _read_text_from_image(roi)
    #     print('[DEBUGGING] text = ', number)

    print('-- [INFO] reading card number by Google vision...')

    # card_number = _read_text_from_roi(card_number_joint_roi)
    # sometimes result of text reading isn't looked nice
    # formated_card_number = _format_card_number(card_number)

    expiry_date = _get_expiry_date(valid_date_roi_arr)
    formated_card_number = '??'
    return formated_card_number


def _prepare_image_for_east_detector(image):
    # getting card number roi

    # The EAST model requires that your input image dimensions be multiples of 32
    # just choose our image size = 320x320
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
            if scoresData[x] < EAST_MIN_CONFIDENCE:
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


def _get_text_boxes(image_for_east, east_image_width, east_image_height, ratio_h, ratio_w):

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    output_layers = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    print("-- [INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(EAST_MODEL_PATH)

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
    (rects, confidences) = _decode_east_text_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    text_roi_arr = []
    # because boundary has not very good accuracy
    # adding padding for better text capturing
    roi_padding_x = 10
    roi_padding_y = 20 # google vision works better if there is some space after roi

    for box in boxes:
        top_left_x = int(box[0] * ratio_w)
        top_left_y = int(box[1] * ratio_h)
        bottom_right_x = int(box[2] * ratio_w)
        bottom_right_y = int(box[3] * ratio_h)

        top_left_x = top_left_x - roi_padding_x
        top_left_y = top_left_y - roi_padding_x
        bottom_right_x = bottom_right_x + roi_padding_x
        bottom_right_y = bottom_right_y + roi_padding_y

        text_roi_arr.append((top_left_x, top_left_y, bottom_right_x, bottom_right_y))



    return text_roi_arr


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


def _get_valid_date_boxes(boxes_list, card_number_text_boxes, image):
    # expiry date box is:
    # -- not card number box (located below)
    # -- side ratio in range 2.6 - 3.5 (was chosen experimentally)
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
            if box_aspect_ratio >= 2.6 and \
               box_aspect_ratio < 3.6:
                valid_date_candidate_boxes.append(box)

    return valid_date_candidate_boxes


def _read_text_from_roi(image):
    request_params = {'key': access_config.GOOGLE_VISION_API_KEY}
    body = make_request(image)
    response = requests.post(url=main_config.VISION_API_URL, params=request_params, json=body)
    response_json = response.json()
    card_number = response_json['responses'][0]['textAnnotations'][0]['description']
    return card_number


def make_request(image):
    image_base_64 = _convert_img_to_base64(image)

    return {
      "requests": [
        {
          "features": [
            {
               "maxResults": 2,
               "type": "DOCUMENT_TEXT_DETECTION"
            }
          ],
          "image": {
             'content': image_base_64
          },
          "imageContext": {
              "languageHints": ["en"]
          }
        }
      ]
    }
    # return  {
    #     "requests": [
    #         {
    #             "features": [
    #                 {
    #                     "maxResults": 50,
    #                     "type": "LANDMARK_DETECTION"
    #                 },
    #                 {
    #                     "maxResults": 50,
    #                     "type": "FACE_DETECTION"
    #                 },
    #                 {
    #                     "maxResults": 50,
    #                     "type": "OBJECT_LOCALIZATION"
    #                 },
    #                 {
    #                     "maxResults": 50,
    #                     "type": "LOGO_DETECTION"
    #                 },
    #                 {
    #                     "maxResults": 50,
    #                     "type": "LABEL_DETECTION"
    #                 },
    #                 {
    #                     "maxResults": 50,
    #                     "type": "DOCUMENT_TEXT_DETECTION"
    #                 },
    #                 {
    #                     "maxResults": 50,
    #                     "type": "SAFE_SEARCH_DETECTION"
    #                 },
    #                 {
    #                     "maxResults": 50,
    #                     "type": "IMAGE_PROPERTIES"
    #                 },
    #                 {
    #                     "maxResults": 50,
    #                     "type": "CROP_HINTS"
    #                 },
    #                 {
    #                     "maxResults": 50,
    #                     "type": "WEB_DETECTION"
    #                 }
    #             ],
    #             "image": {
    #                 "content": image_base_64
    #             },
    #             "imageContext": {
    #                 "cropHintsParams": {
    #                     "aspectRatios": [
    #                         0.8,
    #                         1,
    #                         1.2
    #                     ]
    #                 },
    #                 "languageHints": ["en"]
    #             }
    #         }
    #     ]
    # }


def _convert_img_to_base64(image):
    buffer = cv2.imencode('.jpg', image)[1]
    img_base64 = base64.b64encode(buffer)
    img_base64 = img_base64.decode("utf-8")
    return img_base64


def _format_card_number(card_number):

    card_number_without_spaces = card_number.replace(' ', '')
    print(card_number_without_spaces)
    formated_card_number = card_number_without_spaces[:4] + ' ' + card_number_without_spaces[4:8] + ' ' + \
                           card_number_without_spaces[8:12] + ' ' + card_number_without_spaces[-5:]

    return formated_card_number


def _get_expiry_date(roi_arr):
    # for google api requests reducing
    # joint candidates roi into one roi
    joint_roi = _joint_expiry_date_roi(roi_arr)
    # expiry_date_roi = _define_expiry_date_roi(roi_arr)

    return '??'

from PIL import Image
def _joint_expiry_date_roi(roi_arr):
    # joint_roi_cols = max width among roi + borders_paddings*2 (both side)
    # joint_roi_rows = sum height of all roi + borders_paddings (top side) + joint_roi_space_between_rows * roi_cols
    joint_roi_rows = 0
    joint_roi_cols = 0
    joint_roi_space_between_rows = 20
    joint_roi_space_borders_paddings = 20
    # print('roi_arr ', roi_arr)
    # print('roi_arr shape ', roi_arr.shape)
    for roi in roi_arr:
        roi_w = roi.shape[1]
        roi_h = roi.shape[0]
        print('roi_w = ', roi_w)
        print('roi_h = ', roi_h)
        # cv2.imshow("Debugging date rois", roi)
        # cv2.waitKey(0)

        if roi_w > joint_roi_cols:
            joint_roi_cols = roi_w
        joint_roi_rows = joint_roi_rows + roi_h + joint_roi_space_between_rows
    # cv2.destroyAllWindows()
    joint_roi_rows += joint_roi_space_borders_paddings
    joint_roi_cols = joint_roi_cols + 2*joint_roi_space_borders_paddings

    joint_roi = np.zeros((joint_roi_rows, joint_roi_cols, 3), np.uint8)

    print('joint_roi shape', joint_roi.shape)
    insert_row_start = joint_roi_space_borders_paddings
    insert_col_start = joint_roi_space_borders_paddings
    for idx, roi in enumerate(roi_arr):
        roi_w = roi.shape[1]
        roi_h = roi.shape[0]
        insert_col_end = insert_col_start + roi_w
        insert_row_end = insert_row_start + roi_h

        joint_roi[insert_row_start:insert_row_end, insert_col_start:insert_col_end] = roi
        insert_row_start = insert_row_end + joint_roi_space_between_rows

    # joint_roi[:roi.shape[0], :roi.shape[1]] = roi
    cv2.imshow("Debugging date rois", joint_roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # for roi in roi_arr:



############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "get_roi":
        assert args.image, \
            "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CreditCardConfig()
    else:
        class InferenceConfig(CreditCardConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1


        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.command == "read_card":
        weights_path = LAST_MODEL_WEIGHTS_PATH
    elif args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.command == "read_card":
        model.load_weights(weights_path, by_name=True)
    elif args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "read_card":
        read_card(model, img_file=args.image)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))



