# Custom train Mask RCNN model.

Example of custom training of Mask RCNN model.
Based on work [Waleed Abdulla](https://github.com/matterport/Mask_RCNN).


## How to use

```
# Train a new model starting from pre-trained COCO weights
python3 scripts/train_model.py train --dataset=/path/to/coco/ --model=coco

# Train a new model starting from ImageNet weights
python3 scripts/train_model.py train --dataset=/path/to/coco/ --model=imagenet

# Continue training a model that you had trained earlier
python3 scripts/train_model.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

# Continue training the last model you trained. This will find
# the last trained weights in the model directory.
python3 scripts/train_model.py train --dataset=/path/to/coco/ --model=last

# Split dataset onto train/val
python - m scripts.dataset --action=split_train_val_images


```