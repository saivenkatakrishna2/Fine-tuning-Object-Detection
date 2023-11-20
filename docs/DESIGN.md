### Fine-tuning object detection Design
Authors: Sai venkat krishna Pallampati, Venkat sai Vainala<br>
Date: 11/18/2023

#### AI System Overview:
We use Deep Learning with Transfer Learning for detecting objects with the help of a pre-trained model. 
Fine-tuning object detection: we have to select the dataset from an ethical source and the data in the dataset must be ethically collected.The next step is to annotate the images manually with the help of an opensource  label annotation tool- LabelMe. With the json labels we change the format which is used by the pre-trained model "Detectron model-zoo", we train the model with the images we annotated and then test the model performance.
The purpose of the Deep learning helps the model to detect the exact location of an object in an image.  Transfer learning means using a pre-trained model to train a new model instead of training the new model from the scratch which leads to save the time.<br>
The sources we used in this project are:<br>
For training a model which works for object detection will use GPU to train, so if we don't have a great resource also we can train the model with the help of the Amazon sagemaker- a platfrom which provides services like training and building a high quality machine learning models.<br>
The dataset : We are using the dataset which is an opensource and available in kaggle. We need to know about the opensource tool called **labelme** which is used for annotating images available in GitHub.<br>

Author: Saivenkatakrishna Pallampati<br>
Reviewer: Venkat Sai Vainala