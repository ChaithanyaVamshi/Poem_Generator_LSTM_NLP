# Generating Poems Using Natural Language Processing (NLP) with Python

The Ultimate Guide for Text Generation and Summarization by Predicting the Next Set of Words using LSTM.

![image](https://user-images.githubusercontent.com/31254745/161347627-43023485-fb61-46c2-8c2c-ddae736977b5.png)

## Introduction

Natural Language Processing (NLP) is an emerging field and a subset of machine learning which aims to train computers to understand human languages. The most common application of NLP is Text Generation and Summarization. 

Text generation comes in handy in the world of creative arts through songwriting, poems, short stories, and novels.

In this blog, I will demonstrate and show how we can harness the power of Natural Language Processing (NLP) with Deep Learning by building and training a neural network that generates poems and predicts the next set of words from the seed text using LSTM.

## Steps to Implement Text Generation with LSTM Neural Networks in Python with Keras and NLTK

1.	Problem Statement
2.	Importing Libraries
3.	Loading the Data
4.	Data Pre-processing
5.	Tokenize the Data
6.	Model Building
7.	Model Evaluation Results: Accuracy and Loss
8.	Predicting the next 25 words in a Poem

## 1.	Problem Statement

The objective of this task is that we are given a Gutenberg Dataset and use the book 'blake-poems.txt' file corpus as the source text.

Using this data, we need to train a neural network model that generates poems by predicting the next set of words from the seed text in a Poem using LSTM, Keras and NLTK.

### Why LSTM?

LSTM stands for long short-term memory, and it is a building unit for layers of a recurrent neural network. The LSTM unit is made of a cell, an input, an output and a forget gate.

These are responsible for memorizing over a certain time, where the gates regulate how much of the data is kept and can have a memory about the previous data. Hence it can generate a new pattern using previous data.

LSTMs are mainly used for text generation, and it’s preferred over RNNs because of RNNs vanishing and exploding gradients problems.

## 2.	Importing Libraries

- NLTK: It is a toolkit built for working with NLP in Python which contains text processing libraries for tokenization, parsing, classification etc.
- Keras: It is a powerful and easy-to-use free open-source Python library and high-level API for developing and evaluating deep learning models.

## 3.	Loading the Dataset

- Download the Gutenberg dataset from NLTK
- Print the name of the files in the dataset
- Get the book 'blake-poems.txt' text file from NLTK

## 4.	Data Pre-processing

The pre-processing of the text data is an essential step as it makes it easier to extract information from the text and apply deep learning algorithms to it.

The objective of this step is to Inspect data and clean noise that is irrelevant such as punctuation, special characters, numbers, and terms that don’t carry much weightage in context to the text.

We will perform some basic text pre-processing on the text like Remove punctuations and numbers, Single character removal, removing multiple spaces, limiting text to 5000 etc. so that text is good enough to build the required model on.

## 5.	Tokenization

As we all know machine learning and neural networks algorithms cannot work on text data directly, they simply don’t recognize it. Hence, we need to convert our words in the sentences to numerical values such that our model can figure out what exactly is going on.

For this purpose, especially, in Natural Language Processing (NLP), “Tokenization” plays are vital role. Tokens are individual terms or words, and tokenization is the process of splitting a string of text into tokens. 

Tokenization serves as the base of almost every possible model based on NLP. It just assigns each unique word a different number which we can check out by printing tokenizer.word_index.

## 6.	Model Building

a)	Defining the LSTM Neural Network Model

- we will use an Embedding Layer to learn the representation of words, and a Long Short-Term Memory (LSTM) recurrent neural network to learn to predict words based on their context and we use 50 as the size of the embedding vector space.
- We will use two LSTM hidden layers with 256 memory units each. More memory cells and a deeper network may achieve better results.
- Dropout is a regularization method, where a proportion of nodes in the layer are randomly ignored by setting their weights to zero for each training sample. This technique also improves generalization and reduces overfitting. Hence, to avoid that we will be using a Dropout layer with a probability of 20%.
- The output layer predicts the next word as a single vector the size of the vocabulary with a probability for each word in the vocabulary.
- SoftMax activation function is used to ensure the outputs have the characteristics of normalized probabilities.

b)	LSTM Neural Network Architecture Summary

![image](https://user-images.githubusercontent.com/31254745/161347977-dd50e5d4-b9e7-4b2d-9722-c00d26dded38.png)

c)	Compiling the LSTM Model

Once our layers are added to the model, we need to set up a score function, a loss function, and an optimization algorithm.
- Loss function:  To measure how poorly our model performs on images with known labels and we will use “categorical_crossentropy".
- Optimizer: This function will iteratively improve parameters to minimize the loss. We will go with the "Adam" optimizer.
- Evaluation Metric:  "Accuracy" is used to evaluate the LSTM model's performance.
- Learning rate: The amount of change to the model during each step of this search process, or the step size. The learning rate hyperparameter controls the rate or speed at which the model learns.  We will use 0.001 as the learning rate for the LSTM model.

d)	Model Training 

To start training, we will use the “model.fit” method to train the data and parameters with epochs = 200 and batch_size=64.

We have trained the model using 200 epochs and got a loss of 0.0287 and an accuracy of 100%. Below is the visual summary of the LSTM model Accuracy and Loss.

Results: 

Model Accuracy: 100.00%

Model Loss:  0.02874165028333664

![image](https://user-images.githubusercontent.com/31254745/161348074-aa2337e2-4b06-42bb-bd39-2412ff18b00e.png)

![image](https://user-images.githubusercontent.com/31254745/161348084-45d0f976-fdbc-423d-a9c1-9c3dd989140a.png)

## 7.	Predicting the Next 25 Words of a Poem

Finally, we will input a seed text which will be the origin from which the poem will be generated and set the next words to 25. 

Poem 1:

Seed word sequence gives his light and gives his heat away and flowers and trees and beasts and men receive Comfort in morning joy in the noonday and

Predicted words we are put on earth little space that we may learn to bear the beams of love and these black bodies and this sunburnt face

BLEU Score for predicted words: 1.0

Poem 2:

Seed word sequence gives his light and gives his heat away and flowers and trees and beasts and men receive Comfort in morning joy in the noonday and

Predicted words is but cloud and like shady grove for when our souls have learn the heat to bear The cloud will vanish, we shall hear his

BLEU Score for predicted words: 1.0

Poem 3:

Seed word sequence gives his light and gives his heat away and flowers and trees and beasts and men receive Comfort in morning joy in the noonday and

Predicted words voice saying come out from the grove my love and care and round my golden tent like lambs rejoice thus did my mother say and

BLEU Score for predicted words: 1.0

## 8.	Conclusion

In this blog, we discussed how to implement a neural network that generates poems and predicts the next set of words from the seed text using LSTM, Keras and NLTK.

Though some parts of the poem sound meaningless, the model can be tweaked to gain higher accuracy, lower loss, and predict more meaningful poems.

## 9.	References

- https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/
- https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
