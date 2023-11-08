### Title: Fine-tuning object detection
Authors: Sai venkat krishna Pallampati, Venkat sai Vainala<br>
Date: 11/05/23
#### AI System Types:
We are using "**Deep Learning with Transfer Learning**" because we need to detect object in an image.<br>
Why?<br>
**Deep Learning**:<br>
Object detection and instance segmentation can be done using **Deep learning** provides a fast and accurate means to predict the 'location' of an Object in an Image.
Deep learning is a powerful Machine Learning technique in which the object detector automatically learns image features required for detection tasks.<br>
Several techniques for Object Detection using Deep Learning are available such as `Faster R-CNN`, `you only look once` (YOLO) v2, YOLO v3, YOLO v4, YOLOX, and single shot detection (SSD).
Deep Learning models commonly used for Image Understanding.

**Transfer Learning** : 

Transfer learning, used in Machine learning, is the reuse of a pre-trained model on a new problem. 
In transfer learning, a machine exploits the knowledge gained from a previous task to improve generalization about another. <br> 
In our case we are using `Detectron2's model zoo` as a Pre-trained model in our object detection to not training the model from the scratch.
The main advantages are saving training time and not needing a lot of data. 

Authors : Saivenkatakrishna Pallampati and Venkat Sai Vainala 