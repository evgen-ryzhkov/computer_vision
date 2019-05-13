"""
Prepare image for EAST text detector and getting text boxes

"""

import config.east

import time
import numpy as np
import cv2
from imutils.object_detection import non_max_suppression


class EastTextDetection:

    def get_text_boxes(self, image):

        image_for_east, east_image_width, east_image_height, ratio_h, ratio_w = self._prepare_image_for_east_detector(
            image)

        # define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        output_layers = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

        # load the pre-trained EAST text detector
        print("-- [INFO] loading EAST text detector...")
        net = cv2.dnn.readNet(config.east.EAST_MODEL_PATH)

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
        roi_padding_y = 20  # sometimes google vision works better if there is some space after roi

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
                if scoresData[x] < config.east.EAST_MIN_CONFIDENCE:
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
