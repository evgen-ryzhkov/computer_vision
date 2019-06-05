# Driving licence reader
Scanning Ukrainian driving licence by the camera and recognizing First / Last Name, Birth year, Document ID.

## For what
It could be useful for example:
 - for automatic getting a scan version of the document without a scanner (reducing of cost of the workplace and time for client's service);
 - for automatic getting driver's data for further processing (saving in database for instance);
 - for automatic user registration / authorisation (better user experience). 

## Results examples:
![Ex 1](https://github.com/evgen-ryzhkov/computer_vision/blob/master/driving_licence_reader/images/screens/1.jpg) 
![Ex 2](https://github.com/evgen-ryzhkov/computer_vision/blob/master/driving_licence_reader/images/screens/2.jpg) 
![Ex 3](https://github.com/evgen-ryzhkov/computer_vision/blob/master/driving_licence_reader/images/screens/3.jpg) 


## Pipeline:
![Pipeline](https://github.com/evgen-ryzhkov/computer_vision/blob/master/driving_licence_reader/images/screens/pipeline.png)
1. Detect driving licence (Region of Interest detection) by Mask RCNN (custom trained model).
2. Cut ROI from the image.
3. Preprocessing (rotate and sckew card) for better results of text reading.
4. Detect text boxes by connected component based approach.
5. Define among them driver data rows.
6. Characters recognition by Tesseract OCR.


## How to use:
1. Install requirements (pip install -r requirements.txt)
2. Download custom trained models:
   - [Mask RCNN](https://www.dropbox.com/s/u9ckzuw03tp7fiz/mask_rcnn_driving_licence.h5?dl=0) into /models/mrcnn/
3. Recognize the document by image ( python -m scripts.driving_licence_reader --image=<image file name in images/test directory>)

Notice: With some code updates it may be apply to other types of documents.
