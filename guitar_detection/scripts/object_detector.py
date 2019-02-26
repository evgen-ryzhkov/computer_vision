import config.main as config
import argparse
import cv2
import numpy as np
import time
import imutils


class ObjectDetector:

    def detect_object(self):
        user_arguments = self._get_command_line_arguments()
        file_name = user_arguments['file']

        # define what's type of input file
        point_idx = file_name.find('.')
        ext_len = len(file_name) - point_idx - 1
        file_ext = file_name[-ext_len:]

        if file_ext == 'jpg' or file_ext == 'jpeg' or file_ext == 'png':
            self._detect_on_image(file_name)
        elif file_ext == 'mp4':
            self._detect_on_video(file_name)

    def _detect_on_video(self, file_name):
        out_video = self._define_video_writer()
        net = self._define_neural_network()
        video_stream = self._create_video_stream(file_name)
        object_classes = open(config.OBJECT_CLASSES).read().strip().split('\n')

        self._determine_time_for_result_video_creation(video_stream, net)

        while video_stream.isOpened():
            ret, frame = video_stream.read()

            # if read next frame OK, working with this frame
            if ret == True:
                outs = self._get_list_of_object_boxes_with_probabilites(frame, net)
                boxes, confidences, class_ids = self._get_data_for_visualization(frame, outs)
                strong_boxes_idxs = self._get_idx_for_strong_boxes(boxes, confidences)
                frame_with_detected_objects = self._get_frame_with_detected_objects(strong_boxes_idxs,
                                                                                    boxes,
                                                                                    class_ids,
                                                                                    object_classes,
                                                                                    frame)
                # saving video
                out_video.write(frame_with_detected_objects)

            # if can't read next frame, we reached the end of the video
            else:
                break

        # Release everything if job is finished
        video_stream.release()
        out_video.release()
        cv2.destroyAllWindows()

    def _detect_on_image(self, file_name):
        object_classes = open(config.OBJECT_CLASSES).read().strip().split('\n')
        img_file_path = config.TEST_DIR + file_name

        # load our input image and grab its spatial dimensions
        image = cv2.imread(img_file_path)
        (image_h, image_w) = image.shape[:2]

        net = None
        try:
            net = cv2.dnn.readNetFromDarknet(config.YOLO_CFG_FILE_PATH, config.YOLO_WEIGHTS_PATH)
            print('[INFO] Reading net config - OK')
        except FileNotFoundError:
            print('[ERROR] Check path to YOLO cfg and YOLO weight. Look right values in config/main.py ')

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(self._get_outputs_names(net))

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        class_ids = []

        for out in outs:
            for detection in out:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]  # classes scores starts from index 5
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    # scale the bounding box coordinates back relative to
                    # the size of the frame, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and height
                    box = detection[0:4] * np.array([image_w, image_h, image_w, image_h])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        conf_threshold = 0.5
        nms_threshold = 0.4
        # apply  non-maximum suppression algorithm on the bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # ensure at least one detection exists
        if len(indices) > 0:
            frame = None
            for i in indices:
                i = i[0]
                box = boxes[i]
                box_x = box[0]
                box_y = box[1]
                box_w = box[2]
                box_h = box[3]
                class_id = class_ids[i]
                label = object_classes[class_id]
                frame = self._draw_predict(image, label, box_x, box_y, box_w, box_h)

            cv2.imshow('Resulted video', frame)
            cv2.waitKey(0)

    @staticmethod
    def _get_command_line_arguments():
        ap = argparse.ArgumentParser()
        ap.add_argument("--file", required=True,
                        help="Video file name from data/test_video")
        args = vars(ap.parse_args())
        return args

    @staticmethod
    def _get_outputs_names(net):
        layersNames = net.getLayerNames()
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    @staticmethod
    def _draw_predict(frame, txt, top_left_x, top_left_y, w, h):

        bottom_right_x = top_left_x + w
        bottom_right_y = top_left_y + h

        color = (0, 255, 0)
        cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, 2)
        cv2.putText(frame, txt, (top_left_x, top_left_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    @staticmethod
    def _define_video_writer():
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_video = cv2.VideoWriter(config.OUTPUT_VIDEO_FILE_PATH, fourcc, 20.0,
                                    (640, 360))  # hardcode! must be the same as input video
        return out_video

    @staticmethod
    def _define_neural_network():
        try:
            net = cv2.dnn.readNetFromDarknet(config.YOLO_CFG_FILE_PATH, config.YOLO_WEIGHTS_PATH)
            print('[INFO] Reading net config - OK')
            return net
        except FileNotFoundError:
            print('[ERROR] Check path to YOLO cfg and YOLO weight. Look right values in config/main.py ')

    @staticmethod
    def _create_video_stream(file_name):
        video_file_path = config.TEST_DIR + file_name
        try:
            video_stream = cv2.VideoCapture(video_file_path)
            print('[INFO] Capturing video - OK')
            return video_stream
        except FileNotFoundError:
            print('[ERROR] File {} not found in data/test_video'.format(file_name))

    def _determine_time_for_result_video_creation(self, video_stream, net):
        try:
            prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
                else cv2.CAP_PROP_FRAME_COUNT
            total = int(video_stream.get(prop))
            print("[INFO] {} total frames in video".format(total))

            ret, frame = video_stream.read()
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            # start, end - for calculation the duration of the operation
            start = time.time()
            outs = net.forward(self._get_outputs_names(net))
            end = time.time()
            if total > 0:
                elap = (end - start)
                print("[INFO] single frame took {:.4f} seconds".format(elap))
                print("[INFO] estimated total time to finish: {:.4f}".format(
                    elap * total))

        # an error occurred while trying to determine the total
        # number of frames in the video file
        except:
            print("[INFO] could not determine # of frames in video")
            print("[INFO] no approx. completion time can be provided")
            total = -1

    def _get_list_of_object_boxes_with_probabilites(self, frame, net):
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(self._get_outputs_names(net))
        return outs

    @staticmethod
    def _get_data_for_visualization(frame, outs):
        original_frame_width = frame.shape[1]
        original_frame_height = frame.shape[0]

        # initialize our lists of detected bounding boxes, confidences,
        # and class IDs, respectively
        boxes = []
        confidences = []
        class_ids = []

        for out in outs:
            for detection in out:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                # each detection  has the form like this
                # [center_x center_y width height obj_score class_1_score class_2_score ..]
                scores = detection[5:]  # classes scores starts from index 5
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    # scale the bounding box coordinates back relative to
                    # the size of the frame, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and height
                    box = detection[0:4] * np.array(
                        [original_frame_width, original_frame_height, original_frame_width, original_frame_height])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        return boxes, confidences, class_ids

    @staticmethod
    def _get_idx_for_strong_boxes(boxes, confidences):
        # apply  non-maximum suppression algorithm on the bounding boxes
        # for removing weak boxes
        conf_threshold = 0.5
        nms_threshold = 0.4
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        return indices

    def _get_frame_with_detected_objects(self, strong_boxes_idxs, boxes, class_ids, object_classes, frame):
        for i in strong_boxes_idxs:
            i = i[0]
            box = boxes[i]
            box_x = box[0]
            box_y = box[1]
            box_w = box[2]
            box_h = box[3]
            class_id = class_ids[i]
            label = object_classes[class_id]
            frame = self._draw_predict(frame, label, box_x, box_y, box_w, box_h)
        return frame



# ----------------------------------------------------
od = ObjectDetector()
od.detect_object()