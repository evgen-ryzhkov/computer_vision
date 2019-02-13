import config.main as config
import argparse
import pickle
import cv2
import imutils
import face_recognition


class FaceDetection():

    def __init__(self):
        self.face_encodings_file_path = config.FACE_ENCODINGS_FILE_PATH
        self.test_video_dir_path = config.TEST_VIDEO_DIR
        self.output_video_dir_path = config.OUTPUT_VIDEO_FILE_PATH
        self.default_name = 'Unknown'
        self.confidence_threshold = 20

    def detect_face_on_video(self):
        user_arguments = self._get_command_line_arguments()
        input_video_path = self.test_video_dir_path + user_arguments['video']

        encoding_faces_dict = pickle.loads(open(self.face_encodings_file_path, "rb").read())

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.output_video_dir_path, fourcc, 20.0, (1280, 720)) # hardcode! must be the same as input video

        video_stream = cv2.VideoCapture(input_video_path)

        while video_stream.isOpened():
            ret, frame = video_stream.read()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # resize frame for better perfomance
            rgb = imutils.resize(frame, width=480)

            # for position box correcting on result video
            ratio = frame.shape[1] / float(rgb.shape[1])

            face_boxes, encodings = self._detect_faces_and_encode_them(rgb)

            recognized_character_names = self._get_names_people_on_image(
                known_encoding_faces_dict=encoding_faces_dict,
                encoding_faces_from_image=encodings)
            result_frame = self._show_result(frame, face_boxes, recognized_character_names, ratio)

            # saving video
            out.write(result_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release everything if job is finished
        video_stream.release()
        out.release()
        cv2.destroyAllWindows()

    @staticmethod
    def _get_command_line_arguments():
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--video", required=True,
                        help="Input video file name")
        args = vars(ap.parse_args())
        return args

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

    def _show_result(self, frame, face_boxes, recognized_character_names, ratio):

        for ((top, right, bottom, left), name) in zip(face_boxes, recognized_character_names):
            # rescale the face coordinates
            top = int(top * ratio)
            right = int(right * ratio)
            bottom = int(bottom * ratio)
            left = int(left * ratio)

            # draw the predicted face name on the image
            # if face known - color green, else - red
            color = (0, 255, 0)
            if name == self.default_name:
                color = (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        cv2.imshow('Resulted video', frame)
        return frame



# ----------------------------------------------------
fr_o = FaceDetection()
fr_o.detect_face_on_video()