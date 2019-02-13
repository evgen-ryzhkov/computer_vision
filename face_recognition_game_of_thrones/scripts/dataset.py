import http.client, urllib.parse, json
import config.main as config
import requests
import time
import os
import shutil


class Dataset:

    def __init__(self):
        # todo - replace the key into independent congig file that will be gitignored
        self.subscription_key = 'bfed7294c78b419cb6a46987c125e25b'
        self.search_url = 'https://api.cognitive.microsoft.com/bing/v7.0/images/search'

        self.train_dataset_path = config.TRAIN_IMG_DIR
        self.search_terms = ['jon snow', 'cersei lannister', 'jaime lannister', 'daenerys targaryen']
        self.images_amount_for_class = 100

    def create_train_dataset(self):
        self._clear_directory(self.train_dataset_path)

        for search_term in self.search_terms:
            print('Creating dataset for ', search_term)
            dir_path = self._create_directory(search_term)

            search_results = self._get_json(search_term)
            url_images = gd_o.get_first_n_image_urls(search_results, n_images=self.images_amount_for_class)

            self._parse_and_save_images(url_images, dir_path)

        print('Datasets has been saved successfully.')

    # todo - hardcoding for fast experimenting for increase accuracy
    def create_manually_train_dataset(self):
        search_term = 'robb_stark'
        one_character_train_dataset_path = self.train_dataset_path + '/' + search_term
        self._clear_directory(one_character_train_dataset_path)
        print('Creating dataset for ', search_term)
        search_results = self._get_json(search_term)
        url_images = gd_o.get_first_n_image_urls(search_results, n_images=self.images_amount_for_class)
        self._parse_and_save_images(url_images, one_character_train_dataset_path)
        print('Datasets has been saved successfully.')


    @staticmethod
    def _clear_directory(path):
        for root, dirs, files in os.walk(path):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))

    def _create_directory(self, category_name):
        category_name = category_name.replace(' ', '_')
        category_dir = self.train_dataset_path + '/' + category_name
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
        return category_dir

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

        for idx, img in enumerate(url_images):
            file = requests.get(img, timeout=30)
            file_path = dir_path + '/' + file_name_pattern + str(idx) + file_name_ext

            # write the image to disk
            f = open(file_path, "wb")
            f.write(file.content)
            f.close()

    @staticmethod
    def get_first_n_image_urls(search_results, n_images):
        image_urls = [img["thumbnailUrl"] for img in search_results[:n_images]]
        return image_urls


# ----------------------------------------------------
gd_o = Dataset()
# gd_o.create_train_dataset()
gd_o.create_manually_train_dataset()

