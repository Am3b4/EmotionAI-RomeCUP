# Emotion AI

This project was created for the [ROMECUP 2024](https://romecup.org/)

The Artificial Intelligence detects if the person seen by the camera is smiling or not.

## Creation
First I downloaded two datasets from [Kaggle](https://www.kaggle.com/), than with a python script i moved the files to create a better file structure.
I augmented the data by rotating, flipping and adding noise.
The model i created is a Conv2D with more than 2.700.000 trainable parameters.
I fitted the model for 10 generations and achived a 94% validation accuracy and a loss of 0.25
