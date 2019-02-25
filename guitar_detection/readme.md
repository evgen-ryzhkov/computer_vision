# Face recognition for Game of Thrones  
Face detection of known characters on provided photo. The data set has a few "known" faces.
The others - Unknown. 

![Face recognition for Game of Thrones on image](https://github.com/evgen-ryzhkov/computer_vision/blob/master/face_recognition_game_of_thrones/prtn_screens/screenshot_1.jpg)
 
Tags: Data mining, Bing Image Search API (MS Azure Cognitive service), OpenCV, imutils, dlib, face_recognition.

Scripts:
- dataset.py - creating train and test data set in according to search term

Commands:
- download images: python -m scripts.dataset.py --action create_train_dataset
- split full dataset by train and test:  python -m scripts.dataset.py --action create_train_test_txt
 
 How to use:
 - choose search term for the object, set in dataset.py init
 - load images and del non relevant
 - copy images to other directory (need remove from download directory because
   the directory cleaning before new search images)
- label images in yolo format (with labelImg for example)
- preparing training configuration files
    - create train.txt and test.txt - splited lists of images for traning darknet model
    - copy this files to darknet custom config 
    - modify yolo config for your dataset
- download pretrained convolutional weights
- start training with darknet

