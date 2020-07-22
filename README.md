# Description
Computer animated agents and robots bring new dimension in human computer interaction which makes it vital as how computers can affect our social life in day-to-day activities. Face to face communication is a real-time process operating at a a time scale in the order of milliseconds. The level of uncertainty at this time scale is considerable, making it necessary for humans and machines to rely on sensory rich perceptual primitives rather than slow symbolic inference processes.

In this project we are presenting the real time facial expression recognition of eight most basic human expressions: ANGER, DISGUST, FEAR, HAPPY, NEUTRAL SAD, SURPRISE, PAIN.

This model can be used for prediction of expressions of both still images and real time video. However, in both the cases we have to provide image to the model. In case of real time video the image should be taken at any point in time and feed it to the model for prediction of expression. The system automatically detects face using HAAR cascade then its crops it and resize the image to a specific size and give it to the model for prediction. The model will generate eight probability values corresponding to eight expressions. The highest probability value to the corresponding expression will be the predicted expression for that image.

## Business Problem
We are using the Facial Expression recognition model for predicting the real time condition of the patient in the hospitals. Pain Attribute would give an alert of critical condition of the patient.

For any image our goal is to predict the expression of the face in that image out of eight basic human expression

## Problem Statement
CLASSIFY THE EXPRESSION OF FACE IN IMAGE OUT OF EIGHT BASIC HUMAN EXPRESSION

### Source Data
https://drive.google.com/file/d/13ivfAOOaF8dLtiS-1hKV5uggBLZwl4pt/view?usp=sharing

### Prerequisites
You need to have installed following softwares and libraries in your machine before running this project.  
Python 3  
Anaconda: It will install ipython notebook and most of the libraries which are needed like sklearn, pandas, seaborn, matplotlib, numpy, PIL.  
OpenCV  
keras  
Installing  
Python 3: https://www.python.org/downloads/  
Anaconda: https://www.anaconda.com/download/  
OpenCV: pip install opencv-python  
Keras: pip install keras  
Built With  
ipython-notebook - Python Text Editor  
OpenCV - It is used for processing images  
Keras - Deep Learning Library  
Sklearn: It is a Machine Learning library but here it is used just to calculate accuracy and confusion matrix.  


## Directory setup format
Data->Bottleneck_Features->[Bottleneck_CombinedTrain, Bottleneck_CVHumans, Bottleneck_TestHumans,CombinedTrain_Labels, CVHumans_Labels, TestHumans_Labels],
####
Data->Dataframes,
####
Data->Humans->[Angry, Happy, Sad, Disgust, Pain, Fear, Neutral, Surprise],
####
Data->Model_Save,
####
Data->Test
####
Data->Logs
