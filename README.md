# Deep-learning-Object-detection-using-Tensorflow-API
## Abstract
Efficient and accurate object detection has been an important topic in the advancement of computer vision systems. With the advent of deep learning techniques, the accuracy for object detection has increased drastically. The project aims to incorporate state-ofthe-art technique for object detection with the goal of achieving high accuracy with a real-time performance. A major challenge in many of the object detection systems is the dependency on other computer vision techniques for helping the deep learning based approach, which leads to slow and non-optimal performance. In this project, we use a completely deep learning based approach to solve the problem of object detection in an end-to-end fashion. The network is trained on the most challenging publicly available dataset (Microsoft COCO), on which a object detection challenge is conducted annually. The resulting system is fast and accurate, thus aiding those applications which require object detection.

The TensorFlow Models GitHub repository has a large variety of pre-trained models for various machine learning tasks, and one excellent resource is their object detection API. The object detection API helps us to train our own object detection model for a large variety of different applications. Whether you need a high-speed model to work on live stream high-frames-per-second (fps) applications or high-accuracy desktop models, the API helps us to train and export a model. Here we have used a pre-trained COCO dataset model named ’Faster R-CNN inception v2 COCO’ available in the GitHub repository and performed a fine tuning (retraining) of the model with our own dataset(RGUKT student ID cards) to perform real time object detection when the input is couple of images, video and a live webcam stream.
## 1.1 Problem statement
Many problems in computer vision were saturating on their accuracy before a decade. However, with the rise of deep learning techniques, the accuracy of these problems drastically improved. One of the major problem was that of image classification, which is defined as predicting the class of the image. A slightly complicated problem is that of image localization, where the image contains a single object and the system should predict the class of the location of the object in the image (a bounding box around the object). The more complicated problem (this project), of object detection involves both classification and localization. In this case, the input to the system be an image, video or live webcam stream, the output will be a bounding box corresponding to all the objects in the input, along with the class of object in each box.
## 1.2 Object detection
Object Detection is the process of finding real-world object instances like car, bike, TV, flowers, and humans in still images or Videos. It allows for the recognition, localization, and detection of multiple objects within an image which provides us with a much better understanding of an image as a whole. It is commonly used in applications such as image retrieval, security, surveillance, and advanced driver assistance systems (ADAS).
## 1.3 Tensorflow framework
Tensorflow is Google’s Open Source Machine Learning Framework for dataflow programming across a range of tasks. Nodes in the graph represent mathematical operations, while the graph edges represent the multi-dimensional data arrays (tensors) communicated between them.
<p align="center">
  <img src="doc/pictf.JPG">
</p>

Tensors are just multidimensional arrays, an extension of 2-dimensional tables to data with a higher dimension. There are many features of Tensorflow which makes it appropriate for Deep Learning.

##1.4 Object detection workflow

Object detection workflow has the following steps :

1.Preparation of dataset.

2.Annotating the dataset.

3.Training the model on dataset and performing evaluation.

4.Testing the model on new unseen data.

<p align="center">
  <img src="doc/pic3.jpg">
</p>
