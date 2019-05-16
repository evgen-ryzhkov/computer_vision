# Driving licence reader
Scanning Ukrainian driving licence by the camera and recognizing First / Last Name, Birth year.

## For what
It could be useful for example:
 - for automatic getting a scan version of the document without a scanner (reducing of cost of the workplace and time for client's service);
 - for automatic getting driver's data for further processing (saving in database for instance);
 - for automatic user registration / authorisation (better user experience). 

## Results examples:
![Ex 1](https://github.com/evgen-ryzhkov/computer_vision/blob/master/credit_card_reader/images/screens/1.jpg) 
![Ex 2](https://github.com/evgen-ryzhkov/computer_vision/blob/master/credit_card_reader/images/screens/2.jpg) 
![Ex 3](https://github.com/evgen-ryzhkov/computer_vision/blob/master/credit_card_reader/images/screens/3.jpg) 
![Ex 4](https://github.com/evgen-ryzhkov/computer_vision/blob/master/credit_card_reader/images/screens/4.jpg) 


## Pipeline:
1. Detect driving licence (Region of Interest detection) by Mask RCNN (custom trained model).
2. Cut ROI from the image.
3. Preprocessing (rotate and sckew card) for better results of text reading.
4. Detect text boxes by EAST text detection (custom trained model).
5. Define among them card number box and expiry date candidates.
6. Characters recognition by Google Vision API (it showed not bad results from the box, much better than Tesseract OCR).
7. Card number text formatting and define expiry date among text candidates.

## How to use:
1. Install requirements (pip install -r requirements.txt)
2. Download custom trained models:
   - [Mask RCNN](https://www.dropbox.com/s/g85kkzjpb33ih7u/mask_rcnn_credit_card_last.h5?dl=0) into /models/mrcnn/
   - [EAST](https://www.dropbox.com/s/yvbnme2zvzxc7zp/frozen_custom_east_text_detection.pb?dl=0) into /models/east_text_detector/
3. Recognize card data by image (python -m scripts.card_reader --image=<image file name in images/test directory>)

Notice: It isn't difficult to update scripts for recognition by video.

## Materials:
- [Mask R-CNN for Object Detection and Segmentation](https://github.com/matterport/Mask_RCNN)
- [EAST: An Efficient and Accurate Scene Text Detector](https://github.com/argman/EAST)
- [Google Vision OCR](https://cloud.google.com/vision/docs/ocr)
- Thanks [Adrian Rosebrock](https://www.pyimagesearch.com/) for his explanations and examples. 

