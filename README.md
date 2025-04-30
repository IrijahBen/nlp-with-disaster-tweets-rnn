# nlp-with-disaster-tweets-rnn : https://www.kaggle.com/competitions/nlp-getting-started
# Disaster Tweets Classification with GRUs

This project addresses the challenge of automatically identifying disaster-related tweets from the large volume of daily tweets. It utilizes Natural Language Processing (NLP) techniques and Recurrent Neural Networks (RNNs), specifically Gated Recurrent Units (GRUs), to classify tweets as either reporting a real disaster or using disaster-related terms in a figurative context.

## 1. Introduction

Twitter has become a critical platform for real-time information sharing, especially during disasters. However, distinguishing between genuine emergency reports and irrelevant noise is challenging due to the ambiguity of language. This project aims to build a machine learning model to accurately classify tweets, enabling more effective disaster response.

The model is trained on a dataset of 10,000 hand-classified tweets, labeled as either "disaster" (1) or "not disaster" (0).

## 2.  Modules

The following Python libraries are used in this project:

* `numpy`
* `pandas`
* `re`
* `matplotlib.pyplot`
* `seaborn`
* `sklearn.model_selection`
* `sklearn.metrics`
* `tensorflow`
* `tensorflow.keras.preprocessing.text`
* `tensorflow.keras.preprocessing.sequence`
* `tensorflow.keras.models`
* `tensorflow.keras.layers`
* `tensorflow.keras.callbacks`
* `tensorflow.keras.optimizers`

## 3. Data

The dataset used for training and testing the model is the "Natural Language Processing with Disaster Tweets" dataset available on Kaggle. It consists of tweets and their corresponding labels.

* `train.csv`: Contains the training data with tweet text and target labels.
* `test.csv`: Contains the test data (tweets to be classified).

## 4. Methodology

The project involves the following key steps:

1.  **Data Exploration:** Analyzing the dataset to understand its structure, features, and class distribution.
2.  **Text Preprocessing:** Cleaning and preparing the tweet text for model input, including tokenization, removing noise, and padding sequences.
3.  **Model Building:** Constructing a GRU-based neural network model to classify tweets.
4.  **Model Training:** Training the model on the labeled training data.
5.  **Model Evaluation:** Assessing the model's performance using appropriate metrics.
6.  **Model Tuning:** Optimizing model parameters to improve accuracy.

## 5. Results

The notebook includes a detailed analysis of the model's performance, including classification reports and confusion matrices.

## 6. Future Work

Further improvements could involve:

* Exploring pre-trained word embeddings like Word2Vec.
* Using bidirectional RNNs.
* Employing cross-validation techniques.

## 7. Notebook Contents

The main implementation is in the `nlp-with-disaster-tweets-rnn.ipynb` Jupyter Notebook.
