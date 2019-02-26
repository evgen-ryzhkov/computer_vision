# Custom object detection with OpenCV and YOLO
Guitars detection on image and video files. 

![Guitar detection](https://github.com/evgen-ryzhkov/computer_vision/blob/master/guitar_detection/prtn_screen/custom_object_detection.jpg)
 
Tags: Data mining, Bing Image Search API (MS Azure Cognitive service), OpenCV, YOLO, Darknet.

Scripts:
- dataset.py - creating train and test data set in according to search term
- object_detector.py - guitar detection on image/video file

Commands:
- download images for training: python -m scripts.dataset.py --action create_train_dataset
- split full dataset by train and test:  python -m scripts.dataset.py --action create_train_test_txt
- detect on image: python -m scripts.oject_detector.py --file [image/video file in data/test directory]
 
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
- copy resulted yolo.cfg and yolo.weights to project's model directory
- run object detector 

