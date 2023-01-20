# Machinelearning-Final-Project
The results of the final project in DV2599

DrawAndPredictApp.py should be put in the same folder as the models folder.

- models contains a set of example models for the drawing app

* Processing shows different stages of processing the letters on the drawingboard. 

* Pressing recognise will generate new pictures from the drawing board and put them in processing.


naming convention: 
mod = cnn
mod64 = cnn with 64 batchsize unweighted
mod128w = cnn with batchsize 128 with weights



Make sure to have pickle, tensorflow, and other libraries used in either file installed. 

The datasets can be found on:
https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format (download "A-Z handwritten data.csv")

https://www.kaggle.com/datasets/crawford/emnist?resource=download&select=emnist-balanced-test.csv (download "emnist-byclass train.csv" and rename it to "emnist-byclass.csv")
