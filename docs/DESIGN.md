### Fine-tuning object detection Design
Authors: Sai venkat krishna Pallampati, Venkat sai Vainala<br>
Date: 11/18/2023

#### AI System Overview:
We use Deep Learning with Transfer Learning for detecting objects with the help of a pre-trained model. 
Fine-tuning object detection: we have to select the dataset from an ethical source and the data in the dataset must be ethically collected.The next step is to annotate the images manually with the help of an opensource  label annotation tool- LabelMe. With the json labels we change the format which is used by the pre-trained model "Detectron model-zoo", we train the model with the images we annotated and then test the model performance.
The purpose of the Deep learning helps the model to detect the exact location of an object in an image.  Transfer learning means using a pre-trained model to train a new model instead of training the new model from the scratch which leads to save the time.<br>
The sources we used in this project are:<br>
For training a model which works for object detection will use GPU to train, so if we don't have a great resource also we can train the model with the help of the Amazon sagemaker- a platform which provides services like training and building a high quality machine learning models.<br>
The dataset : We are using the dataset which is an opensource and available in kaggle. We need to know about the opensource tool called **labelme** which is used for annotating images available in GitHub.<br>

Author: Saivenkatakrishna Pallampati<br>
Reviewer: Venkat Sai Vainala

#### Relevant Theoretical Background:
The content knowledge required to understand the fine-tuning object detection are deep learning, transfer learning, understanding bounding boxes, process of Training a new model with the help of pre-trained model. <br>
`Deep learning`: A machine learning technique used for detecting objects. Deep Learning helps to find the exact location of the object in an image. <br>
`Transfer learning`: The process of training a new model with the help of a pre-trained model which trains on a large dataset leads to save the training time and computational resources. <br>
`Bounding Boxes`: It will detect the object in an image with a shape(square). <br>
`Detectron2(Model_zoo)`: To train our model use a pre-trained model `Model_zoo` which is from the detectron2 library(which contains many pre-trained models). <br>
The required modules and frameworks used in the fine-tuning object detection are `Torch`, `Torchvision`, `detectron2`(open source library), `BeautifulSoap`(Python library),
`matplotlib`(for data visualization). Tools required in our project are `labelme` - for annotating images, `amazon sagemaker studio lab` - a jupyterlab IDE used for training
models with the power of AWS, and we can switch from CPU to GPU instance if necessary. We use `kaggle` for searching an open source datasets(a place where we can find
wide range of datasets). <br>

Author: Venkatsai Vainala <br>
Reviewer: Saivenkatakrishna Pallampati

#### AI Development and Evaluation:
The first and foremost thing is to know the concepts regarding the fine-tuning the object detection like `Deep Learning`, `Transfer Learning` and tools like `Labelme` tool. 
And a `Detectron2` framework which we use in our project. The next step is searching a suitable dataset from `Kaggle` labelling the images with the `labelme` tool and after 
labelling the images we get annotations in a json files, Now we will train our model with the json files and the annotated images.The images we train our model is from the open-source dataset `Car License Plate Detection` from kaggle. 
we take some images and annotate them and train our model with those images. We use a pre-trained model `Model_zoo` which is used in our project to detect the objects[Model_zoo is a pre-trained model trained on a larger datasets]. 
We use `Bounding Boxes` to identify our object in the image. And the final step is to evaluate the performance of the model by testing the model with sample test images we can validate the accuracy of the model with the help of `total_loss` and `learning rate`.

Author: Venkatsai Vainala <br>
Reviewer: Saivenkatakrishna Pallampati
