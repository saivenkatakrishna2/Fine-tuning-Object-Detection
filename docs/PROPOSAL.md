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

Authors: Sai Venkata Krishna Pallampati and Venkat Sai Vainala

#### Public AI System:

We are using *"Detectron2 framework"* to investigate the type of AI system
	. It is publicly available resource developed by Facebook AI Research's(FAIR) a next generation library that provides state-of-the-art detection and segmentation algorithms.
	. It is an open-source computer vision framework that embodies the principles of *Deep learning*, particularly relying on the `CNN`
	. It has a rich set of features like pre-trained models and tools that streamline the development process
	. It is also accurate and efficient object recognition and localization within the images.
<br>
We  use transfer learning to '*teach*' the model to detect objects(cars), saving us the Time and Computational resources of training a model from scratch. 
In our case we'll be using the `Mask R-CNN ResNet-50 FPN 3x` model pre-trained for COCO instance segmentation.

Authors: Sai Venkata Krishna Pallampati and Venkat Sai Vainala

#### Theoretical and Practical knowledge on AI:

To complete the project we need to know some concepts like : <br>
**Transfer Learning:** It is the reuse of a pre-trained model on a new problem.  In our case we are using a pretrained model "Detectron2 model zoo" to train our model to detect the Object(car) in an image.<br>
**Model Training**: This is the crucial step in the project, and in this project it  plays a main role in the creation of the model and training of the model and evaluating the performance of the model.<br>
**Detectron2 framework**: It is an open-source computer vision framework which consists many pre-trained models.  In our project we take a pre-trained model named `Detectron2's model zoo` to train our model to detect the cars(objects).<br>
**Labelme**: It is an open source image annotation tool, which is used to create annotations for object detection, classification, and segmentation for computer vision datasets.  We use this tool to label the images in the dataset which helps to train the model.<br>
**Annotating images**: We need to know the process of annotating images using labelme, in which we are using the  images in the dataset.

Author: Sai Venkata Krishna Pallampati <br>
Reviewer: Venkat sai Vainala

#### Resources and Tools Used in AI System:

**Tools**: We have taken the `LabelMe` tool in our AI system to create annotations for the images in our dataset.  
**Platforms**: `Amazon SageMaker Studio Lab` is the platform used in our project. We need to train our model in GPU instance, so we are utilizing this `Amazon SageMaker Studio Lab` to train, test and validate the model.
**Libraries**: We use a framework known as `Detectron2`, where we make use of a pre-trained model from the 'model zoo.' In our project, we use the `Detectron2` framework, which is used for object detection and image analysis. Within this framework, we take a pre-trained model obtained from the `model zoo` to obtain existing knowledge and features for our project. 
**Dataset**: We work with a dataset named `Car License Plate detection` which is taken from `Kaggle`. Our AI system's is focused on car detection, and we apply transfer learning techniques to improve its performance. The dataset is developed by `Larxel`, it contains 433 images but for our project we are going to take 100 images from this dataset and train our model.

Author: Venkat Sai Vainala <br>
Reviewer: Sai Venkata Krishna Pallampati

#### Ethical Considerations for Determining an AI system:

To determine the legitimate use of AI system we need to consider the following properties:<br>
**Accuracy**: The most important property for an AI system is Accuracy, because it needs to produce the outputs accurately and relatively to the inputs provided by the user. So in our project it is one of the most important 
property we are considering to evaluate the model performance.<br>
**Robustness**: Robustness relates to the **Accuracy**.  Our AI system should correctly recognize an input even though the input was adversarial. Model performance can be estimated by this property, we can also assess the accuracy of the model.<br>
**Accountability**: It means Safety and the outcome of the AI system should not harm the society.  Our AI system will detect object, and it doesn't show any bias while detecting the object. 
We need to consider these as the key properties of a legitimate AI system.<br>
**Interpretability**: It means **Usability** in simple words how easy to understand the outcome of the AI system(in our case).  End user is the important for any product, so we should consider them while training a model.<br>
**Ethical**: It includes Privacy, which considers the protection of an Individual identity and data. It also relates to Accountability. In the dataset the data which are using for training should be an ethical manner. The outcomes also must be in ethical manner.

Author: Sai Venkata Krishna Pallampati <br>
Reviewer: Venkat sai Vainala

#### Ethical considerations for developing a safe and trustworthy AI system:

We should consider below concepts for developing a safe and trustworthy AI system.<br>
**Safety**: It means the developed AI system shouldn't cause any kind harm to the end user/society. In our project we are using
ethical data which doesn't harm any end user/society in means of data loss.<br>
**Transparency**: The AI system should be clear while producing the outputs to the user. Our model maintain transparency while developing<br>
**Privacy**: Privacy means protecting the people identity and data.  In our project while developing an AI system we will consider this as the main feature.<br>
**Reliability**: It means Accuracy, whether the system is doing the right thing.  In our case, while developing an AI system we should consider this because we should provide the right output accurately.

Author: Venkat Sai Vainala <br>
Reviewer: Sai Venkata Krishna Pallampati

#### References:

“Getting Started with Object Detection Using Deep Learning - MATLAB & Simulink.” Accessed November 7, 2023. <br>
https://www.mathworks.com/help/vision/ug/getting-started-with-object-detection-using-deep-learning.html.

“What Is Transfer Learning? A Guide for Deep Learning | Built In.” Accessed November 7, 2023. <br>
https://builtin.com/data-science/transfer-learning.

"LabelMe tool" <br>
http://labelme.csail.mit.edu/Release3.0/ <br>
https://github.com/wkentaro/labelme

"Detectron2 Model Zoo and Baselines" <br>
https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md

"Detectron2" <br>
https://github.com/facebookresearch/detectron2/tree/main

"Car License Plate Detection" <br>
https://www.kaggle.com/datasets/andrewmvd/car-plate-detection

Authors: Venkat sai Vainala and Sai Venkata Krishna Pallampati
