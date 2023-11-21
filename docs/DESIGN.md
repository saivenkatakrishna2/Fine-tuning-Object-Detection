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

#### Project Milestones and Deliverables
The divided project plan for each week is given below:

| Week1                                                                                                                                                                                           | Week2                                                                                                                                           | Week3                                                                                                                                                                                                   | Week4                                                                                                                                                          | Week5                                                                                                                                                      |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Task 1- finding dataset, writing proposal, writing design, gathering required frameworks and models for fine-tuning.                                          Author: Venkat Sai Vainala        | Task 1- labelling images in dataset       Author: Venkat Sai Vainala                                                                            | Task 1-  Environment setup, code implementation.                                                    Author: Venkat Sai Vainala                                                                          | Task 1- Evaluating model performance with  our dataset.                                                          Author: Venkat Sai Vainala                    | Task 1- PPT Preparation for demo, document preparation.                                                                        Author: Venkat Sai Vainala  |
| Task 2- finding dataset, writing proposal, writing design, reviewing frameworks and models for fine-tuning.                                                Author: Saivenkatakrishna Pallampati | Task 2-- labelling images in dataset.                                                                      Author: Saivenkatakrishna Pallampati | Task 2- dataset setup and gathering annotated images required to train the model, code implementation.                                                             Author: Saivenkatakrishna Pallampati | Task 2- fixing Errors and finalizing report.                                                                              Author: Saivenkatakrishna Pallampati | Task 2- Demo preparation, document preparation.                                                                       Author: Saivenkatakrishna Pallampati |


Authors: Venkat Sai Vainala and Saivenkatakrishna Pallampati

#### References:
References:
“Getting Started with Object Detection Using Deep Learning - MATLAB & Simulink.” Accessed November
7, 2023.<br>
https://www.mathworks.com/help/vision/ug/getting-started-with-object-detection-using-deeplearning.html. <br>
“What Is Transfer Learning? A Guide for Deep Learning | Built In.” Accessed November 7, 2023. <br>
https://builtin.com/data-science/transfer-learning. <br>
"LabelMe tool"<br>
http://labelme.csail.mit.edu/Release3.0/ <br>
https://github.com/wkentaro/labelme <br>
"Detectron2 Model Zoo and Baselines" <br>
https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md <br>
"Detectron2" <br>
https://github.com/facebookresearch/detectron2/tree/main <br>
"Car License Plate Detection" <br>
https://www.kaggle.com/datasets/andrewmvd/car-plate-detection <br>
AWS. 2023. “What Is Deep Learning? - Deep Learning Explained - AWS.” Amazon Web Services, Inc. 2023. <br> 
https://aws.amazon.com/what-is/deep-learning/. <br>
"Machine Learning Service - Amazon SageMaker" <br>
https://aws.amazon.com/pm/sagemaker/?gclid=Cj0KCQiApOyqBhDlARIsAGfnyMqLF8yVUkzroO0bt8UlygBdIaY5KuE_63F_DfWTcYXHVt77HSPl3X4aAjAQEALw_wcB&trk=b6c2fafb-22b1-4a97-a2f7-7e4ab2c7aa28&sc_channel=ps&ef_id=Cj0KCQiApOyqBhDlARIsAGfnyMqLF8yVUkzroO0bt8UlygBdIaY5KuE_63F_DfWTcYXHVt77HSPl3X4aAjAQEALw_wcB:G:s&s_kwcid=AL!4422!3!651751060698!p!!g!!amazon%20sagemaker%20studio!19852662230!145019226177 <br>
https://studiolab.sagemaker.aws/ <br>
Authors: Saivenkatakrishna Pallampati 