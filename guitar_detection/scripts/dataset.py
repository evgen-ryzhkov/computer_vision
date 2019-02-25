import http.client, urllib.parse, json
import config.main as config
import config.access as config_access
import requests
import time
import os
import shutil
import argparse
from imutils import paths
import random



class Dataset:

    def __init__(self):
        self.subscription_key = config_access.AZURE_SUBSCRIPTION_KEY
        self.search_url = 'https://api.cognitive.microsoft.com/bing/v7.0/images/search'

        self.train_dataset_path = config.TRAIN_IMG_DIR
        self.search_term = ['kiss concert'] # for diversity use different term 'guitar', 'iron maiden concert' and so on
        self.images_amount_for_class = 300

    def run(self):
        try:
            user_arguments = self._get_command_line_arguments()

            if user_arguments['action'] == 'create_train_dataset':
                self._create_train_dataset()
            elif user_arguments['action'] == 'create_train_test_txt':
                self._create_train_test_txt()
            else:
                print('Incorrect action type. You should choose one of possible values.')
        except:
            print('Argument --action is required. Possible values: "create_train_dataset", "create_train_test_txt"')

    @staticmethod
    def _get_command_line_arguments():
        ap = argparse.ArgumentParser()
        ap.add_argument("--action", required=True,
                        help="What's action you need? Possible values: 'create_train_dataset', 'create_train_test_txt' ")
        args = vars(ap.parse_args())
        return args

    def _create_train_dataset(self):
        print('[INFO] Creating dataset...')
        self._clear_directory(self.train_dataset_path)
        search_results = self._get_json(self.search_term)

        url_images = gd_o.get_first_n_image_urls(search_results, n_images=self.images_amount_for_class)
        self._parse_and_save_images(url_images, self.train_dataset_path)

        print('[OK] Datasets has been saved successfully.')

    @staticmethod
    def _clear_directory(path):
        for root, dirs, files in os.walk(path):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))

    def _get_json(self, search_term):
        headers = {"Ocp-Apim-Subscription-Key": self.subscription_key}
        offset = 0
        n_grab_images = 0
        value_results = []

        while n_grab_images < self.images_amount_for_class:
            params = {"q": search_term, "imageType": "photo", "offset": offset}
            response = requests.get(self.search_url, headers=headers, params=params)
            response.raise_for_status()
            search_results = response.json()

            value_field = search_results['value']
            n_grab_images += len(value_field)
            offset = search_results['nextOffset']
            value_results += value_field

            # print(json.dumps(search_results, indent=4, sort_keys=True))
            # There is a limit on requests per sec for free azure account
            time.sleep(1)

        return value_results

    @staticmethod
    def _parse_and_save_images(url_images, dir_path):

        file_name_pattern = 'img_'
        file_name_ext = '.jpg'
        start_idx = 400 # for increasing amount of dataset. take last number of image from current dataset

        for idx, img in enumerate(url_images):
            file = requests.get(img, timeout=30)
            file_name = file_name_pattern + str(start_idx + idx) + file_name_ext
            file_path = dir_path + file_name
            print('[INFO] saving ' + file_name)

            # write the image to disk
            f = open(file_path, "wb")
            f.write(file.content)
            f.close()

    @staticmethod
    def get_first_n_image_urls(search_results, n_images):
        image_urls = [img["contentUrl"] for img in search_results[:n_images]] # using big image as on small can't see guitars
        return image_urls

    @staticmethod
    def _create_train_test_txt():
        print('[INFO] Creating train and test.txt...')

        list_labeled_image_paths = list(paths.list_images(config.LABELED_TRAIN_IMG_DIR))
        n_full_dataset = len(list_labeled_image_paths)
        random.Random(4).shuffle(list_labeled_image_paths) # shuffle list for better model learning and simplier split data by train / test

        test_size = 0.2
        n_test_size = round(n_full_dataset * test_size)
        n_train_size = n_full_dataset - n_test_size
        train_images_paths = list_labeled_image_paths[:n_train_size]
        test_images_paths = list_labeled_image_paths[-n_test_size:]

        # creating and saving txt files for darknet training
        # we need full path to images in order to darknet could fild them
        file_train = open(config.TXT_TRAIN_FILE, 'w')
        for img in train_images_paths:
            file_train.write(os.path.abspath(img) + '\n')

        file_test = open(config.TXT_TEST_FILE, 'w')
        for img in test_images_paths:
            file_test.write(os.path.abspath(img) + '\n')

        print('[OK] ' + config.TXT_TRAIN_FILE + ' and ' + config.TXT_TEST_FILE + ' have been created successfully.')


# ----------------------------------------------------
gd_o = Dataset()
gd_o.run()
