# Downloading the data

## Complete Image Dataset
Since we used the 2017 COCO dataset, which has around 118,000 training images, 5,000 validation images and 40,000 test images, we decided to give instructions on how to obtain the dataset rather than including it. We first tried multiple python libraries to obtain COCO, such as `fiftyone`, but we found those to be slow and less reliable.

Instead, we simply pulled the dataset from the internet at the following links:
 - http://images.cocodataset.org/zips/train2017.zip
 - http://images.cocodataset.org/zips/val2017.zip
 - http://images.cocodataset.org/zips/test2017.zip
 - http://images.cocodataset.org/annotations/annotations_trainval2017.zip

Then, we unzipped the directories and annotation file using the `unzip` command. We include the whole bash script below and attach a bash file to this directory.
```bash
# Pulling the images and annotations
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Unzip the files
unzip train2017.zip
unzip val2017.zip
unzip test2017.zip
unzip annotations_trainval2017.zip
```

## Small Image Dataset
In the process of debugging our model, we tried various approaches and, in particular, overfitting on a small portion of the full dataset. The set of such images(8 of them) can be seen in the `data\small_train2017` directory. This is also the set of untampered images we used to make predictions for each models for the sake of comparison. You can access the bounding box predictions on these images in the `results` directory, where we run such predictions for each model including the non overfitting ones.
