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

## 1.4 Object detection workflow

Object detection workflow has the following steps :

1.Preparation of dataset.

2.Annotating the dataset.

3.Training the model on dataset and performing evaluation.

4.Testing the model on new unseen data.

<p align="center">
  <img src="doc/pic3.jpg">
                                     
                                       FIGURE 1.2:  Object detection workflow
  
  ## 1.5 Applications of Object detection
  ### 1.5.1 Facial recognition
  A deep learning facial recognition system called the “DeepFace” has been developed by a group of researchers in the Facebook, which identifies human faces in a digital image very effectively. Google uses its own facial recognition system in Google Photos, which automatically segregates all the photos based on the person in the image. There are various components involved in Facial Recognition like the eyes, nose, mouth and the eyebrows.
  <p align="center">
  <img src="doc/pic4.jpg">
  
                                             FIGURE 1.3:  Facial recognition
 
  ### 1.5.2 People counting
  Object detection can be also used for people counting, it is used for analyzing store performance or crowd statistics during festivals. These tend to be more difficult as people move out of the frame quickly.
It is a very important application, as during crowd gathering this feature can be used for multiple purposes.
  <p align="center">
  <img src="doc/pic5.jpg">
  
                                             FIGURE 1.4: People Counting

  ### 1.5.3 Industrial quality check
   Object detection is also used in industrial processes to identify products. Finding a specific object through visual inspection is a basic task that is involved in multiple industrial processes like sorting, inventory management, machining, quality management, packaging etc.
Inventory management can be very tricky as items are hard to track in real time. Automatic object counting and localization allows improving inventory accuracy.
  <p align="center">
  <img src="doc/pic6.jpg">
  
                                            FIGURE 1.5: Industrial quality check
                                            
  ### 1.5.4 Self driving car
  Self-driving cars are the Future, there’s no doubt in that. But the working behind it is very tricky as it combines a variety of techniques to perceive their surroundings, including radar, laser light, GPS, odometry, and computer vision.
  Advanced control systems interpret sensory information to identify appropriate navigation paths, as well as obstacles and once the image sensor detects any sign of a living being in its path, it automatically stops. This happens at a very fast rate and is a big step towards Driverless Cars.
  <p align="center">
  <img src="doc/car.JPG">
  
  ### 1.5.5 Security
  Object Detection plays a very important role in Security. Be it face ID of Apple or the retina scan used in all the sci-fi movies.
It is also used by the government to access the security feed and match it with their existing database to find any criminals or to detect the robbers’ vehicle.
The applications are limitless.

## 2. Faster Region-based Convolutional Network (Faster RCNN)
Region proposals detected with the selective search method were still necessary in the previous model, which is computationally expensive. S. Ren and al. (2016) have introduced Region Proposal Network (RPN) to directly generate region proposals, predict bounding boxes and detect objects. The Faster Region-based Convolutional Network (Faster RCNN) is a combination between the RPN and the Fast R-CNN model.

A CNN model takes as input the entire image and produces feature maps. A window of size 3x3 slides all the feature maps and outputs a features vector linked to two fullyconnected layers, one for box-regression and one for box-classification. Multiple region proposals are predicted by the fully-connected layers. A maximum of k regions is fixed thus the output of the box-regression layer has a size of 4k (coordinates of the boxes, their height and width) and the output of the box-classification layer a size of 2k (“objectness” scores to detect an object or not in the box). The k region proposals detected by the sliding window are called anchors.
<p align="center">
  <img src="doc/pic8.jpg">
                                 
                                 FIGURE 2.1: Faster Region-based Convolutional Network
 
When the anchor boxes are detected, they are selected by applying a threshold over the “objectness” score to keep only the relevant boxes. These anchor boxes and the feature maps computed by the initial CNN model feeds a Fast R-CNN model.

Faster R-CNN uses RPN to avoid the selective search method, it accelerates the training and testing processes, and improve the performances. The RPN uses a pre-trained model over the ImageNet dataset for classification and it is fine-tuned on the PASCAL VOC dataset. Then the generated region proposals with anchor boxes are used to train the Fast R-CNN. This process is iterative.

The best Faster R-CNNs have obtained mAP scores of 78.8% over the 2007 PASCAL VOC test dataset and 75.9% over the 2012 PASCAL VOC test dataset. They have been trained with PASCAL VOC and COCO datasets. One of these models is 34 times faster than the Fast R-CNN using the selective search method.

<p align="center">
  <img src="FasterRcnn.jpg">
</p>

## 3. Building the Object detection classifier
  ### 3.1 Installation of required software
  The system should have anaconda software version 3.0. We have been using windows 8.0 OS and we have installed anaconda 3-5.2.0 x64 bit. Python version v3.7.1 should also be installed. One can install the python package using the following command in Anaconda prompt.   "conda install -c anaconda python".
  ### 3.1.1 Dependencies
  Tensorflow Object Detection API depends on the following libraries:
•	Tensorflow
•	Numpy
•	Matplotlib
•	Pillow 1.0
•	lxml
•	tf Slim
•	Protobuf 3.0.0.

A typical user can install Tensorflow using following command, Installing tensorflow will install all the above mentioned libraries Protobuf, pillow, lxml, tfslim, Numpy and Matplotlib. As our system does not has any GPU such as Nvidia, Radeon etc., we will go with Tensorflow CPU version intsallation by issuing the follwing command in Anaconda prompt.
Command for installing Tensorflow CPU version:

"pip install tensorflow"

The above installations are needed for image detection using Tensorflow API.
For object detection in a live webcam stream using Tensorflow API, we need an extra module called openCV version 3.0. OpenCV module allows us to initialize a camera object and to read the frame from the live webcam stream so that we can detect objects. One can install openCV using Anaconda prompt using the following command 

"conda install -c conda-forge opencv"
  ### 3.1.2 Set up TensorFlow Directory and Anaconda Virtual Environment
  The TensorFlow Object Detection API requires using the specific directory structure provided in its GitHub repository. It also requires several additional Python packages, specific additions to the PATH and PYTHONPATH variables, and a few extra setup commands to get everything set up to run or train an object detection model.
  ### 3.1.3 Download the Faster-RCNN-Inception-V2-COCO model from TensorFlow’s model zoo
  TensorFlow provides several object detection models (pre-trained classifiers with specific neural network architectures) in its model zoo. Some models (such as the SSD-MobileNet model) have an architecture that allows for faster detection but with less accuracy, while some models (such as the Faster-RCNN model) give slower detection but with more accuracy. We initially started with the SSD-MobileNet-V1 model, but it did not do a very good job identifying the ID cards in my images. We re-trained our detector on the FasterRCNN-Inception-V2 model, and the detection worked considerably better, but with a noticeably slower speed.
  
You can choose which model to train your objection detection classifier on. If you are planning on using the object detector on a device with low computational power (such as a smart phone or Raspberry Pi), use the SDD-MobileNet model. If you will be running your detector on a decently powered laptop or desktop PC, use one of the RCNN models.

Here we will use the Faster-RCNN-Inception-V2 model. Download the model here. Open the downloaded faster rcnn inception v2 coco 2018 01 28.tar.gz file with a file archiver such as WinZip and extract the faster rcnn inception v2 coco 2018 01 28 folder to the C drive/tensorflow1/models/research/object detection folder.
### 3.1.4 Set up new Anaconda virtual environment
Next, we will work on setting up a virtual environment in Anaconda for tensorflow-cpu. From the Start menu in Windows, search for the Anaconda Prompt utility and run it.
In the command terminal that pops up, create a new virtual environment called “tensorflow1” by issuing the following command :
 
C:\> conda create -n tensorflow1 pip python=3.5
 
Then, activate the environment by issuing :
 
C:\> conda activate tensorflow1
 
Install tensorflow-cpu in this environment by issuing :
 
(tensorflow1) C:\> pip install tensorflow
 
Install the other necessary packages by issuing the following commands :

(tensorflow1) C:\> conda install -c anaconda protobuf, 
(tensorflow1) C:\> pip install pillow, 
(tensorflow1) C:\> pip install lxml, 
(tensorflow1) C:\> pip install Cython, 
(tensorflow1) C:\> pip install jupyter, 
(tensorflow1) C:\> pip install matplotlib, 
(tensorflow1) C:\> pip install pandas, 
(tensorflow1) C:\> pip install opencv-python.

### 3.1.5 Configure PYTHONPATH environment variable
A PYTHONPATH variable must be created that points to the /models, /models/research, and /models/research/slim directories. Do this by issuing the following commands :
 
(tensorflow1) C:\> set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research; C:\tensorflow1\models\research\slim
 
(Note: Every time the ”tensorflow1” virtual environment is exited, the PYTHONPATH variable is reset and needs to be set up again.)
### 3.1.6 Compile Protobufs and run setup.py
Next, compile the Protobuf files, which are used by TensorFlow to configure model and training parameters. Unfortunately, the short protoc compilation command posted on TensorFlow’s Object Detection API installation page does not work on Windows. Every .proto file in the /object detection/protos directory must be called out individually by the command.

In the Anaconda Command Prompt, change directories to the /models/research directory and copy and paste the following command into the command line and press Enter :

 
protoc --python_out=. .\object_detection\protos\anchor_generator.proto .
\object_detection\protos\argmax_matcher.proto .\object_detection\protos \bipartite_matcher.proto .\object_detection\protos\box_coder.proto .
\object_detection\protos\box_predictor.proto .\object_detection\protos
\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection
\protos\faster_rcnn_box_coder.proto .\object_detection\protos
\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto . \object_detection\protos\image_resizer.proto .\object_detection\protos \input_reader.proto .\object_detection\protos\losses.proto .\object_detection \protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .
\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .
\object_detection\protos\pipeline.proto .\object_detection\protos
\post_processing.proto .\object_detection\protos\preprocessor.proto . \object_detection\protos\region_similarity_calculator.proto .\object_detection \protos\square_box_coder.proto .\object_detection\protos\ssd.proto .
\object_detection\protos\ssd_anchor_generator.proto .\object_detection
\protos\string_int_label_map.proto .\object_detection\protos\train.proto .
\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos
\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto
 
This creates a name pb2.py file from every name.proto file in the /object detection/protos folder.
(Note: TensorFlow occassionally adds new .proto files to the /protos folder. If you get an error saying ImportError: cannot import name ’something something pb2’ , you may need to update the protoc command to include the new .proto files.)
Finally, run the following commands from the C:/tensorflow1/models/research directory one after the other:
 
(tensorflow1) C:\tensorflow1\models\research> python setup.py build

(tensorflow1) C:\tensorflow1\models\research> python setup.py install

## 4. Data set preparation
Now that the TensorFlow Object Detection API is all set up and ready to go, we need to provide the images it will use to train a new detection classifier.
  ### 4.1 Gather Pictures
  TensorFlow needs hundreds of images of an object to train a good detection classifier. To train a robust classifier, the training images should have random objects in the image along with the desired objects, and should have a variety of backgrounds and lighting conditions. There should be some images where the desired object is partially obscured, overlapped with something else, or only halfway in the picture.
  
  For our RGUKT student ID card detector, we have three different objects we want to detect (the ID label Manikanta, Saikumar and Pradeep). We used our SONY digital camera to take about 476 pictures of cards on its own, with various other non-desired objects and multiple cards in the picture. We know we want to be able to detect the cards when they are overlapping, so we made sure to have the cards be overlapped in many images.
 
<p align="center">
  <img src="doc/pic9.jpg">
  
                                        FIGURE 4.1.1: IIIT student ID card Dataset

Make sure the images are not too large. They should be less than 300KB each, and their resolution should not be more than 720x1280. The larger the images are, the longer it will take to train the classifier. After we have all the pictures we need, we have moved 20% of them to the /object detection/images/test directory, and 80% of them to the /object detection /images/train directory. We made sure that there are a variety of pictures in both the /test and /train directories.
  ### 4.2 Label Pictures
  With all the pictures gathered, it is time to label the desired objects in every picture. LabelImg is a great tool for labeling images, and its GitHub page has very clear instructions on how to install and use it. We downloaded and installed LabelImg, point it to our /images/train directory, and then drawn a box around each object in each image. Repeated the process for all the images in the /images/test directory.
<p align="center">
  <img src="doc/pic10.jpg">
  
                                    FIGURE 4.2.1 : labelImg Annotation tool

LabelImg saves a .xml file containing the label data for each image. These .xml files will be used to generate TFRecords, which are one of the inputs to the TensorFlow trainer. Once you have labeled and saved each image, there will be one .xml file for each image in the /test and /train directories.
  ### 4.3 Generate Training Data
  With the images labeled, it is time to generate the TFRecords that serve as input data to the TensorFlow training model. First, the image .xml data will be used to create .csv files containing all the data for the train and test images. From the /object detection folder, issue the following command in the Anaconda command prompt:
 
(tensorflow1) C:\tensorflow1\models\research\object_detection> python xml_to_csv.py
 
This creates a train labels.csv and test labels.csv file in the /object detection/images folder. Now we have to gnerate TF record files which serve as input to the training pipeline. Generate the TFRecord files by issuing these commands from the /object detection folder:
 
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images
\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images
\test --output_path=test.record
 
These generate a train.record and a test.record file in /object detection. These will be used to train the new object detection classifier.
  ### 4.4 Create Label Map and Configure Training
  The last thing to do before training is to create a label map and edit the training configuration file.
  #### Label map
  Label map
The label map tells the trainer what each object is by defining a mapping of class names to class ID numbers. Use a text editor to create a new file and save it as labelmap.pbtxt in the C:/tensorflow1/models/research/object detection/training folder. (Make sure the file type is .pbtxt, not .txt)
 
item {

id: 1 name: ’Manikanta’

}

item {

id: 2 name: ’Saikumar’

}

item {

id: 3 name: ’Pradeep’

}

 The label map ID numbers should be the same as what is defined in the generate tfrecord.py file
 #### Configure training
 Finally, the object detection training pipeline must be configured. It defines which model and what parameters will be used for training. This is the last step before running training.
Navigate to C:/tensorflow1/models/research/object detection/samples/configs and copy the faster rcnn inception v2 pets.config file into the /object detection/training directory. Then, open the file with a text editor. There are several changes to make to the .config file, mainly changing the number of classes and examples, and adding the file paths to the training data.
Make the following changes to the faster rcnn inception v2 pets.config file. Note: The paths must be entered with single forward slashes (NOT backslashes), or TensorFlow will give a file path error when trying to train the model Also, the paths must be in double quotation marks ( ” ), not single quotation marks ( ’ ).
Line 9. Changed num classes to the number of different objects we want the classifier to detect. For our project Manikanta, Sakumar, and Pradeep, it would be num classes : 3 Line 110, Change fine tune checkpoint to:
fine tune checkpoint :
 
"C:/tensorflow1/models/research/object_detection/faster_rcnn_inception_v2_coco_2018 _01_28/model.ckpt"
 
Lines 126 and 128, In the train input reader section, change input path and label map path to:
 
input_path : "C:/tensorflow1/models/research/object_detection/train.record"
 
label map path:
 
"C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"
 
Line 132, Change num examples to the number of images you have in the /images/test directory. In our case it was 102, i.e., having 102 test images in the test folder. Lines 140 and 142, In the eval input reader section, change input path and label map path to: input path :
 
"C:/tensorflow1/models/research/object_detection/test.record"
 
label map path:
 
"C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"
 
Save the file after the changes have been made. That is it, The training job is all configured and ready to go.

 ### 4.5 Run the Training
 From the /object detection directory, issue the following command to begin training:
 
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training
/faster_rcnn_inception_v2_pets.config
 
If everything has been set up correctly, TensorFlow will initialize the training. The initialization can take up to 80 seconds before the actual training begins. When training goes on, it will look like this:
<p align="center">
  <img src="doc/pic11.jpg">
                                   
                                   FIGURE 4.5.1: Object detection classifier Training window
 
Each step of training reports the loss. It will start high and get lower and lower as training progresses. For our training on the Faster-RCNN-Inception-V2 model, it started at about 3.0 and quickly dropped below 1.50. We allowed our model to train until the loss consistently drops below 0.05, which took about 13000 steps, or about 60 hours (depending on how powerful your CPU and GPU are). Note: The loss numbers will be different if a different model is used. MobileNet-SSD starts with a loss of about 20, and should be trained until the loss is consistently under 2.
### 4.6 Export Inference Graph
After the loss is consistently below 0.05 we can stop the training process by issuning command ”ctrl c” in the Anaconda prompt. Now that training is complete, the last step is to generate the frozen inference graph (.pb file). From the /object detection folder, issue the following command, where “XXXX” in “model.ckpt-XXXX” should be replaced with the highest-numbered .ckpt file in the training folder:
 
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training
/model.ckpt-12639 --output_directory inference_graph
 
This creates a frozen inference graph.pb file in the /object detection/inference graph folder.
The .pb file contains the object detection classifier.

## 5. Results and Simulations
  ### Testing the classifier
  The object detection classifier is all ready to go. We have written Python scripts to test it out on an image, video, or webcam feed.
Due to the fact that most of the laptops have 0.9 Megapixels front facing webcamera, So we must not use it for feeding input images because it produces images of very poor quality and our classifier fails in detecting objects from the low quality feed.
Hence, we come up with an idea of using our smartphone rear camera as the webcamera to the laptop, So that we would have better quality input image feed and better results. For this purpose we have used a software called ”DroidCam” in both smartphone and laptop which helps us to connect phone camera to the laptop using a common wifi hotspot.

To test our object detector, move a picture of the object or objects into the /object detection folder, and change the ’image name’ variable in the Object detection image.py to match the file name of the picture. Alternatively, we can use a video of the objects (using Object detection video.py), or just plug in a USB webcam (or DroidCam client) and point it at the objects (using Object detection webcam.py).

To run any of the scripts, type “idle” in the Anaconda Command Prompt (with the “tensorflow1” virtual environment activated) and press ENTER. This will open IDLE, and from there, we can open any of the scripts and run them.
If everything is working properly, the object detector will initialize for about 10 seconds and then display a window showing any objects it is detected in the image.
  ### Results
  
<p align="center">
  <img src="doc/pic12.jpg">
  
<p align="center">
  <img src="doc/pic13.jpg">
 
<p align="center">
  <img src="doc/pic14.jpg">

<p align="center">
  <img src="doc/pic15.jpg">
 
<p align="center">
  <img src="doc/pic16.jpg">
  
