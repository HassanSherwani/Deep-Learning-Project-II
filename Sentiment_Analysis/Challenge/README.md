# Challenge
This is the code for Siraj's challenge on Sentiment Analysis for Intro to Deep Learning.

# Problem statement

Predict the sentiment in game titles from the IGN game dataset. Classification outputs a class from the set of 11 classes. Classes: ('Great', 'Good', 'Okay', 'Mediocre', 'Amazing', 'Bad', 'Awful', 'Painful', 'Unbearable', 'Masterpiece')

Performing sentiment analysis on a small number of output classes such as 2 or 3 (Positive, Negative, Neutral) would give us much higher accuracies compared to using these 11 classes. The task of analyzing the sentiment at a very fine granular level is a hard task.

# Notebooks:

There are two notebooks added

- One that applied dense neural network using TensorFlow
- Second notebook applies same problem with keras.

One might wonder what is the difference in these two techniques. Please check following amazing reading

https://medium.com/implodinggradients/tensorflow-or-keras-which-one-should-i-learn-5dd7fa3f9ca0


Modules
pandas <br>
scikit-learn<br>
tflearn<br>
keras<br>
nltk<br>
numpy<br>



Datasets
Game sentiment prediction - IGN Dataset. 

I have created two datasets out of main one. One has key features "text" and "sentiment". Other one (as was asked in challenge) contains "text" and "score_phrase".
