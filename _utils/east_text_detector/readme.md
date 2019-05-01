# EAST: An Efficient and Accurate Scene Text Detector

### Introduction
[EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155v2).
[Link to repository](https://github.com/argman/EAST)

### Utilites

Copy images from download directory to renamed (temp) directory with names format
for east training requirements (img_[number]) for further labaling
```
python dataset.py --action=rename_downloaded
```

Convert PascalVoc annotation from LabelImg to to icdar_2015 format for EAST model training
```
python dataset.py --action=convert_annotation
```

Create .pb file (it needs for opencv for instance)
```
python freeze_model.py
```

