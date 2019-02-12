# creating base of known faces as a dictionary of pairs of
# face encoding (128-d vector) and character name

import config.main as config
from imutils import paths
import imutils
import cv2
import face_recognition
import os
import pickle


class KnownFaces:

    def __init__(self):
        self.train_dir_path = config.TRAIN_IMG_DIR
        self.face_encodings_file_path = config.FACE_ENCODINGS_FILE_PATH

    def create_base(self):
        known_encodings, known_names = self._get_face_encodings_and_names()
        self._save_dict_face_encodings(known_encodings, known_names)

    def _get_face_encodings_and_names(self):
        train_image_paths = list(paths.list_images(self.train_dir_path))

        known_encodings = []
        known_names = []

        for (idx, imagePath) in enumerate(train_image_paths):
            # extract the person name from the image path
            print("[INFO] processing image {}/{}".format(idx + 1,
                                                         len(train_image_paths)))
            name_from_path = imagePath.split(os.path.sep)[-2]
            character_name = name_from_path.replace('_', ' ').title()

            # load the input image and convert it from BGR (OpenCV ordering)
            # to dlib ordering (RGB)
            image_brg = cv2.imread(imagePath)

            # in order to avoid out of memory
            image_height, image_width, image_channels = image_brg.shape
            if image_width > 1000:
                image_brg = imutils.resize(image_brg, width=1000)

            rgb = cv2.cvtColor(image_brg, cv2.COLOR_BGR2RGB)

            # detect the (x, y)-coordinates of the bounding boxes
            # corresponding to each face in the input image
            boxes = face_recognition.face_locations(rgb, model='cnn')

            # compute the facial embedding for the face
            encodings = face_recognition.face_encodings(rgb, boxes)

            # loop over the encodings
            for encoding in encodings:
                # add each encoding + name to our set of known names and
                # encodings
                known_encodings.append(encoding)
                known_names.append(character_name)

        return known_encodings, known_names

    def _save_dict_face_encodings(self, encodings_list, names_list):
        # dump the facial encodings + names to disk
        print("[INFO] serializing encodings...")
        data = {"encodings": encodings_list, "names": names_list}
        f = open(self.face_encodings_file_path, "wb")
        f.write(pickle.dumps(data))
        f.close()
        print('[OK] Encoding faces dict was saved successfully.')


# ----------------------------------------------------
cfd_ob = KnownFaces()
cfd_ob.create_base()
