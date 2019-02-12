import config.main as config
import argparse
import pickle
import cv2
import imutils
import face_recognition


class FaceDetection():

    def __init__(self):
        self.face_encodings_file_path = config.FACE_ENCODINGS_FILE_PATH
        self.test_images_dir_path = config.TEST_IMG_DIR
        self.default_name = 'Unknown'
        self.confidence_threshold = 10

    def recognize_on_image(self):
        user_arguments = self._get_command_line_arguments()
        input_img_path = self.test_images_dir_path + user_arguments['image']
        print("[INFO] recognizing faces...")
        encoding_faces_dict = pickle.loads(open(self.face_encodings_file_path, "rb").read())
        image_brg, image_rgb = self._get_processed_image(input_img_path)

        face_boxes, encodings = self._detect_faces_and_encode_them(image_rgb)

        recognized_character_names = self._get_names_people_on_image(
            known_encoding_faces_dict=  encoding_faces_dict,
            encoding_faces_from_image=  encodings)
        self._show_resulted_image(image_brg, face_boxes, recognized_character_names)

    @staticmethod
    def _get_command_line_arguments():
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--image", required=True,
                        help="Input image name")
        args = vars(ap.parse_args())
        return args

    @staticmethod
    def _get_processed_image(img_path):
        image_brg = cv2.imread(img_path)

        # in order to avoid out of memory
        image_height, image_width, image_channels = image_brg.shape
        if image_width > 1000:
            image_brg = imutils.resize(image_brg, width=1000)

        # convert to RGB format for dlib
        image_rgb = cv2.cvtColor(image_brg, cv2.COLOR_BGR2RGB)

        return image_brg, image_rgb

    @staticmethod
    def _detect_faces_and_encode_them(image_rgb):
        # detect the (x, y)-coordinates of the bounding boxes corresponding
        # to each face in the input image, then compute the facial embeddings
        # for each face

        boxes = face_recognition.face_locations(image_rgb, model='cnn')
        encodings = face_recognition.face_encodings(image_rgb, boxes)
        return boxes, encodings

    def _get_names_people_on_image(self, known_encoding_faces_dict, encoding_faces_from_image):

        detected_names = []

        # loop over the facial embeddings
        for encoding in encoding_faces_from_image:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(known_encoding_faces_dict["encodings"], encoding)
            display_name = self.default_name

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, item) in enumerate(matches) if item]

                counts = {}
                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = known_encoding_faces_dict["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # show counts of matches for debugging
                print(counts) # for debugging
                # determine the recognized face with the largest number of votes
                max_count_name = max(counts, key=counts.get)

                # in order to be sure that we found known face
                if counts[max_count_name] >= self.confidence_threshold:
                    display_name = max_count_name
                else:
                    display_name = self.default_name

            detected_names.append(display_name)

        return detected_names

    def _show_resulted_image(self, image_brg, face_boxes, recognized_character_names):

        for ((top, right, bottom, left), name) in zip(face_boxes, recognized_character_names):
            # draw the predicted face name on the image
            # if face known - color green, else - red
            color = (0, 255, 0)
            if name == self.default_name:
                color = (0, 0, 255)

            cv2.rectangle(image_brg, (left, top), (right, bottom), color, 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(image_brg, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        cv2.imshow('Resulted image', image_brg)
        cv2.waitKey(0)


# ----------------------------------------------------
fr_o = FaceDetection()
fr_o.recognize_on_image()
