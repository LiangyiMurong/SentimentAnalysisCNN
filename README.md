# Convolutional Neural Network(CNN) - Sentiment Analysis for Customer Reviews in Chinese

## Abstract

Convolutional Neural Network(CNN) has
been applied in many natural language
processing tasks including sentiment analysis,
and sentence classification. The model has
also achieved some state-of-art performance
in some tasks and has been proved to be fast
to conduct. For this project, I trained the CNN
model on a dataset about customer’s reviews
on a product in Chinese and predict Whether
a customers review is positive or negative. By
analyzing the reviews, convolutional neural
networks (CNN) trained on top of pre-trained
word vectors is used in this project to predict
whether the review express a sense of negative
attitude or positive attitude. Comparing to logistic
regression, The CNN models discussed
herein improve the performance of sentiment
prediction and achieve an f1-score of 85.37.

## Problem Definition and Data

Using the review as text data, the problem is to find
out what sentiment each review show in the text.
Each of the terms in the review are represented by
a word vector from a pre-trained word2vec embedding
matrix. By using CNN, the model is expected
to learn the important features within the review
that result in making the judgment. For example,
term like ’good’ in English should probably result
in a positive sentiment because it shows a positive
attitude.
The dataset for this project keep tracks of the
271360 negative reviews and 115600 positive reviews,
386960 reviews in total. Among these data,
the reviews are relatively short and precise so they
are easy to process. To feed the data into the models,
the data has to be transformed into the format
with labels and text.

## Methodology

### Preprocessing the data
1. **Stopwords Removal**: 
Stop words such as ’the’,
’is’ in English don’t have informative value for
making the judgment, so I remove these words
similar to them in Chinese from the corpus to
improve accuracy.

2. **Word2vec**: The word2vec model is trained on
Chinese Wikipedia data, and I construct a word
embedding matrix that contains only the word
in the vocabulary of the training dataset. The
word2vec matrix has a dimension of 100.

3. **Indexing and padding**: The reviews are mapped
into vector representation with reviews as rows
and Vocabulary as columns. Each term in the
reviews are indexed as its position index in the
vocabulary. In addition, Reviews may have
different length with each other, so I have to pad
each review into same length for CNN filters scan.

### Logistic Regression

For logistic regression, I created a sparse vectorized
matrix for the reviews and use the gradient
descent to iteratively nd the best parameter .
This model is based on the assumption that the
frequency of words and their appearance are
correlated with sense of the review. Intuitive
enough, some words are more likely to appear in
positive reviews while others are more likely to
appear in negative reviews, so logistic regression
is one of the simplest yet efficient way to conduct
sentiment analysis. With a sampling size of
150000, the logistic regression model will take
about 1.5 hours to finish, which is even more
time-consuming for larger dataset.

### Convolutional Neural Network

The hyperparameters are referenced and adjusted
from Yoon Kim’s paper: RELU, lter windows(h)
of 8 with 300 feature maps each, dropout rate (p)
of 0.5, l2 constraint (s) of 3, and mini-batch size
of 50.[1] With a filter of kernel size 8, the CNN
model can perceive the relationship of terms in
context and identify the most important features
to make the judgment. By convolving the reviewwordvector
matrix, we get a convolutional layer.
The convolutional layer is then mapped to a pooling
layer through an activation method (RELU
here). By leaving the maximum value of the pooling
layer, we get a single value(vector) for each
convolutional layer, and we can get our final output
by performing a softmax to values(vectors)
from different convolutional layers. This CNN
model is a single layer model with word embeddings
fixed over the training process.

## Evaluation and Results

After preprocessing the data, the whole data set
are stored as a list of dictionaries identifying:
1) the label(1 for positive review and 0 for
negative review), 2) text(tokenized word list), 3)
class(train, dev, test), 4) length of the text. The
ratio between train data, development data and
test data is set to be 0.8:0.1:0.1. For training
and testing, I can set the sampling number for
randomly getting the size of whole data I want
from the dataset, and use them to improve my
models. The baseline I set for this task is a random
guess that achieves a f1 score of 58.45. After
training and testing, my logistic regression model
achieve a f1 score of 80.36, and this score will
have some fluctuation each time because I have
to limit the sampling size due to the restriction of
ram. CNN, on the other hand, achieve a f1 score
of 85.37 with the sampling size of 30000, which
is much higher than logistic regression, and it
indicates the superiority in CNN in this task.
Also, I believe that one restriction for this model
is the word2vec model built only with small
corpus of text. The resources for pre-trained
word2vec matrix are rare so I have to train the
word2vec on my own using the dataset. However,
training a well-developed word2vec can certainly
help to improve performance because the word
embeddings would be more precise.

| Model        | F1           | Training Size  |
| ------------- |:-------------:| -----:|
| Baseline1: Random Guess     | 58.45 | 386960 |
| Baseline2: Logistic Regression     | 80.36      |   30000 |
| Convolutional Neural Network | 85.37      |    30000 |

## Reference

[1] Kim, Yoon. 2014. Convolutional neural networks for
sentence classification. arXiv preprint arXiv:1408.5882 .

[2] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey
Dean. 2013. Efficient estimation of word representations
in vector space. ICLR Workshop.

[3] Andrew L. Maas, Raymond E. Daly, Peter T. Pham,
Dan Huang, Andrew Y. Ng, and Christopher Potts. 2011.
Learning word vectors for sentiment analysis. In Proceedings
of the 49th Annual Meeting of the Association for
Computational Linguistics: Human Language Technologies,
 Volume 1 (HLT ’11), Vol. 1. Association for Computational
Linguistics, Stroudsburg, PA, USA, 142-150.

[4] Maite Taboada , Julian Brooke , Milan Tofiloski ,
Kimberly Voll , Manfred Stede, Lexicon-based methods for
sentiment analysis, Computational Linguistics, v.37 n.2,
p.267-307, June 2011.

[5] OKeefe T, Koprinska I. Feature selection and weighting
methods in sentiment analysis. In: Proceedings of the 14th
Australasian document computing symposium, Sydney,
Australia, ACL; 2009.

[6] Zhang, L., Ghosh, R., Dekhil, M., Hsu, M., and
Liu, B. 2011. Combining lexicon-based and learning-based
methods for Twitter sentiment analysis. Technical Report
HPL-2011-89.
