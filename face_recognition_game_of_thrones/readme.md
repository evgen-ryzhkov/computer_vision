# Face recognition for Game of Thrones  
Face detection of known characters on provided photo. The data set has a few "known" faces.
The others - Unknown. 

![Face recognition for Game of Thrones on image](https://github.com/evgen-ryzhkov/computer_vision/blob/master/face_recognition_game_of_thrones/prtn_screens/screenshot_1.jpg)
 
Tags: Data mining, Bing Image Search API (MS Azure Cognitive service), OpenCV, imutils, dlib, face_recognition.

Scripts:
- dataset.py - creating train data set of character faces that are listed in init of the class
- known_faces.py - creating base of known faces as a dictionary of pairs of face encoding (128-d vector) and character name
- face_detection.py - face recognition on provided photo
- face_detection_video_file.py - face recognition on provided video file

Commands:
- face detection image: python -m scripts.face_detection.py --image [file_name_in data/test dir]
- face detection video: python -m scripts.face_detection_video_file.py --video [file_name_in data/test video dir]
 
