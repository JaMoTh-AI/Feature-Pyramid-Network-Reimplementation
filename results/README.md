# Model Results
This directory contains all of the model results for each of the models that we trained. Each model includes visualizations 
of bounding box predictions for the 8 images contained in the small overfitting training dataset. There are also training loss 
graphs for each of the models.

## Built In Detectron2 Model
![SampleImage](built_in_detectron2/visualized_000000184613.jpg)

## Custom Bottom Up Backbone
### Full Dataset
![SampleImage](custom_bottom_up/visualized_000000391895.jpg)


## ResNet34 Backbone
### Full Dataset
![SampleImage](resnet34/full/visualized_000000060623.jpg)

### Small Dataset
![SampleImage](resnet34/small/visualized_000000318219.jpg)


## ResNet34 Backbone with Pretrained RPN and Fast RCNN Weights
### Small Dataset
![SampleImage](resnet34_pretrained_weights/small/visualized_000000522418.jpg)


## ResNet34 Backbone w/Pretrained Weights and Last Level Pooling Layer
### Full Dataset
![SampleImage](resnet34_pretrained_weights_pooling/full/visualized_000000391895.jpg)

### Small Dataset
![SampleImage](resnet34_pretrained_weights_pooling/small/visualized_000000574769.jpg)

