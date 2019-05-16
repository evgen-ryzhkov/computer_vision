"""
Functions:
-- getting biggest contour
-- getting Canny parameters automatically
-- getting vertices
-- getting bird eye view
-- converting image into base4

"""

import cv2
import imutils
import numpy as np
import base64


class ImageProcessing:

    def get_birds_eye_view_roi(self, instance_image):
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

    @staticmethod
    def convert_img_to_base64(image):
        buffer = cv2.imencode('.jpg', image)[1]
        img_base64 = base64.b64encode(buffer)
        img_base64 = img_base64.decode("utf-8")
        return img_base64
