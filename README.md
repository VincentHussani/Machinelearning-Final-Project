# Machinelearning-Final-Project
The results of the final project in DV2599

The notebook and DrawAndPredictApp.py should be put in the same folder. Create two folders where those files are called "processing" and "models". 
Both of those folders will be included in the zip with some example models and content. 
Processing shows different stages of processing the letters on the drawingboard. 
Pressing recognise will generate new pictures from the drawing board and put them in processing.

naming convention: 
mod = cnn
mod64 = cnn with 64 batchsize unweighted
mod128w = cnn with batchsize 128 with weights



Make sure to have pickle, tensorflow, and other libraries used in either file installed. 

The datasets can be found on:
https://www.kaggle.com/datasets/sachinpatel21/az-handwritten-alphabets-in-csv-format (download "A-Z handwritten data.csv")
https://www.kaggle.com/datasets/crawford/emnist?resource=download&select=emnist-balanced-test.csv (download "emnist-byclass train.csv" and rename it to "emnist-byclass.csv")
