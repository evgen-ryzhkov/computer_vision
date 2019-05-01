"""
Work with datasets

Written by Evgeniy Ryzhkov

------------------------------------------------------------

Usage:

    # Copy images from download directory to renamed (temp) directory with names format
      for east training requirements (img_[number])
    python dataset.py --action=rename_downloaded

    # Convert PascalVoc annotation from LabelImg to to icdar_2015 format for
      EAST model training
    python dataset.py --action=convert_annotation


"""

import os
import shutil
import argparse
import xml.etree.ElementTree as ET


class Dataset:

    def __init__(self):
        self.downloaded_images_path = 'datasets/downloaded'
        self.renamed_images_path = 'datasets/renamed'
        self.file_name_root = 'img_'
        self.train_dataset_path = 'datasets/train'

    def run(self):
        try:
            user_arguments = self._get_command_line_arguments()

            if user_arguments['action'] == 'rename_downloaded':
                self._rename_downloaded_images()
            elif user_arguments['action'] == 'convert_annotation':
                self._convert_annotations_pascalvoc_to_icdar_2015()
            else:
                print('Incorrect action type. You should choose one of possible values.')
        except ValueError:
            print('Argument --action is required. Possible values: "download_images"' + str(ValueError))

    @staticmethod
    def _get_command_line_arguments():
        ap = argparse.ArgumentParser()
        ap.add_argument("--action", required=True,
                        help="What's action you need? Possible values: 'download_images', 'split_train_val_images' ")
        args = vars(ap.parse_args())
        return args

    def _rename_downloaded_images(self):
        self._clear_directory(self.renamed_images_path)

        files_amount = len(os.listdir(self.downloaded_images_path))

        for idx, file in enumerate(os.listdir(self.downloaded_images_path)):
            try:
                file_path = os.path.join(self.downloaded_images_path, file)
                shutil.copy(file_path, self.renamed_images_path)

                # rename file
                old_file = os.path.join(self.renamed_images_path, file)
                old_file_name, file_extension = os.path.splitext(old_file)
                new_file = os.path.join(self.renamed_images_path, self.file_name_root + str(idx) + file_extension)
                os.rename(old_file, new_file)
            except ValueError:
                print('[ERROR] File copy error ' +  str(ValueError))

        print('[OK] {} files have been successfully copied.'.format(files_amount))

    @staticmethod
    def _clear_directory(path):
        for root, dirs, files in os.walk(path):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))

    def _convert_annotations_pascalvoc_to_icdar_2015(self):

        annotation_files_amount = 0
        for file in os.listdir(self.train_dataset_path):
            if file.endswith(".xml"):
                annotation_files_amount+=1

                file_icdar_2015_string = ''
                file_path = os.path.join(self.train_dataset_path, file)
                root = ET.parse(file_path).getroot()
                for object in root.findall('object'):
                    name = object.find('name').text
                    xmin = object.find('bndbox/xmin').text
                    ymin = object.find('bndbox/ymin').text
                    xmax = object.find('bndbox/xmax').text
                    ymax = object.find('bndbox/ymax').text

                    # icdar_2015 format x_top_left, y_top_left, x_top_right, y_top_right, x_bottom_right, y_bottom_right, x_bottom_left, y_bottom_left, text_value
                    x_top_left = xmin
                    y_top_left = ymin
                    x_top_right = xmax
                    y_top_right = ymin
                    x_bottom_right = xmax
                    y_bottom_right = ymax
                    x_bottom_left = xmin
                    y_bottom_left = ymax
                    object_icdar_2015_string = '{},{},{},{},{},{},{},{},{}\n'.format(x_top_left, y_top_left, x_top_right, y_top_right, x_bottom_right, y_bottom_right, x_bottom_left, y_bottom_left, name)
                    file_icdar_2015_string += object_icdar_2015_string

                # saving icdar_annotation to txt file with the same name as xml
                xml_file_name, file_extension = os.path.splitext(file)
                txt_file_name = xml_file_name + '.txt'
                txt_file_path = os.path.join(self.train_dataset_path, txt_file_name)

                txt_file = open(txt_file_path, "w")
                txt_file.write(file_icdar_2015_string)
                txt_file.close()

        print('[OK] {} files have been successfully converted.'.format(annotation_files_amount))









# ----------------------------------------------------
gd_o = Dataset()
gd_o.run()
