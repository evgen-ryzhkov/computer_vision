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

    # Apply color splash to an image
    python object_detector.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>

    # Debugging
    python object_detector.py get_roi --image=<image file name in images/test directory>

"""


# import config.main as config

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

import cv2
import imutils


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

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mrcnn/mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Path to test images dir
TEST_IMAGES_DIR = os.path.join(ROOT_DIR, "images/test/")

# Path to last trained weights
LAST_MODEL_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mrcnn/mask_rcnn_credit_card_last.h5")


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
            image = skimage.io.imread(image_path)
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
                epochs=35,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


def get_credit_card_roi(model, img_file=None):
    image = cv2.imread(TEST_IMAGES_DIR + img_file)
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

        # getting masked roi
        credit_card_instance = masked_img[roi_box[0]:roi_box[2], roi_box[1]:roi_box[3]]

        # look for debugging
        # cv2.imshow("Masked roi", credit_card_instance)
        # cv2.waitKey(0)

        instance_with_backround_around = add_background_around_instance_roi(credit_card_instance)

        bird_eye_view_instance = get_birds_eye_view_roi(instance_with_backround_around)

        cv2.imshow("Debugging", bird_eye_view_instance)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



        return credit_card_instance

    else:
        print('Credit cards were not found.')


def add_background_around_instance_roi(instance_img):
    # for good detecting external contour
    # it's required to be empty space around the object
    # so we create a little bit bigger image from instance with filled background around

    # Create black blank image
    instance_img_height = instance_img.shape[0]
    instance_img_width = instance_img.shape[1]

    instance_with_background_around_height = instance_img_height + 10
    instance_with_background_around_width = instance_img_width + 10
    instance_with_background_around = np.zeros((instance_with_background_around_height, instance_with_background_around_width, 3), np.uint8)
    instance_with_background_around[:] = (0, 0, 0) # black background

    instance_with_background_around[4:instance_img_height+4, 4:instance_img_width+4] = instance_img


    # cv2.imshow("Instance with background around", instance_with_background_around)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return instance_with_background_around


def get_birds_eye_view_roi(instance_image):
    # image = cv2.imread(TEST_IMAGES_DIR + img_file)

    # for increase work speed
    # maybe it will need to turn on
    # ratio = image.shape[0] / 300.0
    # image = imutils.resize(image, height=300)

    biggest_contour = get_biggest_contour(instance_image)
    vertices = get_vertices(biggest_contour)

    birds_eye_view_image = get_birds_eye_view_image(instance_image, vertices)
    return birds_eye_view_image


def get_biggest_contour(image):
    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)

    # find the biggest contours in the edged image
    card_contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest_contour = max(card_contours, key=cv2.contourArea)

    # debugging if the contour is right
    # cv2.drawContours(image, [biggest_contour], -1, (0, 255, 0), 3)
    # cv2.imshow("The biggest contour", image)
    # cv2.waitKey(0)

    return biggest_contour


def get_vertices(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box


def get_birds_eye_view_image(image, four_points):
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

    # look for debugging
    # cv2.imshow("The vertices", birds_eye_view_image)
    # cv2.waitKey(0)
    return birds_eye_view_image


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
    elif args.command == "splash":
        assert args.image or args.video, \
            "Provide --image or --video to apply color splash"
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
    if args.command == "get_roi":
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
    if (args.command == "get_roi") or (args.command == "rotate"):
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
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    elif args.command == "get_roi":
        get_credit_card_roi(model, img_file=args.image)

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))



