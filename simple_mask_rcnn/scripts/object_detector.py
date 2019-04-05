# Lesson from https://www.pyimagesearch.com/2018/11/19/mask-r-cnn-with-opencv/
# and https://www.learnopencv.com/deep-learning-based-object-detection-and-instance-segmentation-using-mask-r-cnn-in-opencv-python-c/

import config.main as config
import argparse
import cv2
import numpy as np


class CocoObjectDetector:

    def __init__(self):
        # Initialize the model parameters
        self.conf_threshold = 0.5
        self.mask_threshold = 0.3

        # Give the textGraph and weight files
        self.text_graph = config.MODEL_GRAPH_FILE
        self.model_weights = config.MODEL_WEIGHTS_FILE
        self.object_classes = self._get_object_classes()
        self.mask_colors = self._get_mask_colors()

    def detect_objects(self):
        user_arguments = self._get_command_line_arguments()
        file_name = user_arguments['file']
        image_path = config.DEMO_DIR + file_name
        self._detect_on_image(image_path)

    @staticmethod
    def _get_command_line_arguments():
        ap = argparse.ArgumentParser()
        ap.add_argument("--file", required=True,
                        help="File name from data/demo_images")
        args = vars(ap.parse_args())
        return args

    def _detect_on_image(self, img_path):
        # Load the network
        net = cv2.dnn.readNetFromTensorflow(self.model_weights, self.text_graph)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        image = cv2.imread(img_path)
        # check if image has been read correctly
        try:
            image_h, image_w = image.shape[:2]
        except AttributeError:
            print('[ERROR] Image file not found!')
            return

        blob = cv2.dnn.blobFromImage(image, swapRB=True, crop=False)
        net.setInput(blob)

        # Run the forward pass to get output from the output layers
        boxes, masks = net.forward(['detection_out_final', 'detection_masks'])

        numClasses = masks.shape[1]
        num_detections = boxes.shape[2]

        # loop over the number of detected objects
        for i in range(num_detections):
            # extract the class ID of the detection along with the confidence
            # (i.e., probability) associated with the prediction
            class_id = int(boxes[0, 0, i, 1])
            confidence = boxes[0, 0, i, 2]

            # filter out weak predictions by ensuring the detected probability
            # is greater than the minimum probability
            if confidence > self.conf_threshold:

                # scale the bounding box coordinates back relative to the
                # size of the image and then compute the width and the height
                # of the bounding box
                box = boxes[0, 0, i, 3:7] * np.array([image_w, image_h, image_w, image_h])
                (startX, startY, endX, endY) = box.astype("int")
                box_w = endX - startX
                box_h = endY - startY

                # extract the pixel-wise segmentation for the object, resize
                # the mask such that it's the same dimensions of the bounding
                # box, and then finally threshold to create a *binary* mask
                mask = masks[i, class_id]
                mask = cv2.resize(mask, (box_w, box_h), interpolation=cv2.INTER_CUBIC)
                mask = (mask > self.mask_threshold)

                # getting ROI
                roi = image[startY:endY, startX:endX]
                cv2.imshow('ROI', roi)
                cv2.waitKey(0)

                # getting instance
                visMask = (mask * 255).astype("uint8")
                instance = cv2.bitwise_and(roi, roi, mask=visMask)

                cv2.imshow('Instance', instance)
                cv2.waitKey(0)

                # now, extract *only* the masked region of the ROI by passing
                # in the boolean mask array as our slice condition
                roi = roi[mask]

                color = np.array([0.0, 255.0, 0.0])
                blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

                # store the blended ROI in the original image
                image[startY:endY, startX:endX][mask] = blended

                cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)



        cv2.imshow('Resulted image', image)
        cv2.waitKey(0)

    @staticmethod
    def _get_object_classes():
        classes = None
        with open(config.CLASS_NAMES_FILE_PATH, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')
        return classes

    @staticmethod
    def _get_mask_colors():
        with open(config.MASK_COLORS_FILE_PATH, 'rt') as f:
            colorsStr = f.read().rstrip('\n').split('\n')
        colors = []  # [0,0,0]
        for i in range(len(colorsStr)):
            rgb = colorsStr[i].split(' ')
            color = np.array([float(rgb[0]), float(rgb[1]), float(rgb[2])])
            colors.append(color)
        return colors


# ----------------------------------------------------
od = CocoObjectDetector()
od.detect_objects()
