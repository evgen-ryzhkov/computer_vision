# Face recognition for Game of Thrones  
Face detection of known characters on provided photo. The data set has a few "known" faces.
The others - Unknown. 

![Face recognition for Game of Thrones on image](https://raw.githubusercontent.com/username/projectname/branch/path/to/img.png)
 
Tags: Bing Image Search API (MS Azure Cognitive service), Deep learning, CNN, OpenCV, imutils, dlib, face_recognition.

Scripts:
- dataset.py - creating train data set of character faces that are listed in init of the class
- known_faces.py - creating base of known faces as a dictionary of pairs of face encoding (128-d vector) and character name
- face_detection.py - face recognition on provided photo

Commands:
- face detection: python -m scripts.face_detection --image [file_name_in data/test dir]
 
