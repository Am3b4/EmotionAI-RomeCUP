# Emotion AI

This project was created for the [ROMECUP 2024](https://romecup.org/)

The Artificial Intelligence detects if the person seen by the camera is smiling or not.

## Creation
First I downloaded two datasets from [Kaggle](https://www.kaggle.com/), than with a python script i moved the files to create a better file structure.
I augmented the data by rotating, flipping and adding noise.
The model i created is a Conv2D with more than 2.700.000 trainable parameters.
I fitted the model for 10 generations and achived a 94% validation accuracy and a loss of 0.25

The dataset on which the Ai was trained contained the entire spectrum of emotions (sadness, anger, fear etc.). Because of the difference in the amount of data labeld with different emotions was different, the data was augmented: different types of noise were added, images were rotated on the x and y axes, until the label distribution was uniform. 


Later I traned a model with 260.000 paremeters (10 times less than the first model) and achived an accuracy of 93.7 on validation.
Also mananged to run it on the GPU via WSL (it was a nightmare), now training and inference are much faster, enabeling a 30 fps live demo, and training is 3-4 times faster
