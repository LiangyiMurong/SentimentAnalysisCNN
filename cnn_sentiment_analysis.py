import numpy as np
import math
from sklearn.metrics import f1_score
import jieba
import re
from collections import defaultdict
import operator
import time
from gensim.models import word2vec
import pickle
import argparse
import random

from keras import backend
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.constraints import unitnorm
from keras.regularizers import l2
from sklearn.metrics import roc_auc_score

# Reference: Yoon Kim (2014): https://github.com/yoonkim/CNN_sentence
# Reference: Sentiment Analysis with Convolutional Networks: https://github.com/vsl9/Sentiment-Analysis-with-Convolutional-Networks

class Data_Processing(object):
	""" Data processing; Returning: 1)processed dataset, 2)vocabulary, 3)word2vec matrix  """

	''' 
	Open the raw data file (*.txt)
	Split the corpus into seperate lines
	Clean the data that doesn't make sense
	'''

	def load_data(self, filename):
	    f = open(filename, 'r', encoding='utf-8')
	    examples = f.read().split('\n')
	    examples = list(filter(lambda a:a != 'na', examples))
	    f.close()
	    return examples

	''' 
	Process the sentences in data file and tokenize them
	Tag each sentence with a dict indentifying its: 1) y(pos or neg), 2) text(tokenized list), 3) class(train, dev, test), 4) len(length of the line)
	'''

	def get_data_and_labels(self, neg_fname, pos_fname, override = False, tokenize = False):
		# Check if local files can be loaded
		if override:
		    info_list = pickle.load(open('dataset.p',"rb"))
		    vocab = pickle.load(open('vocab.p',"rb"))

		# Process the raw data and save them into local files
		else:
			vocab = defaultdict(float)
			info_list = []

			neg = self.load_data(neg_fname)
			pos = self.load_data(pos_fname)

			# Using this stopwords list will not improve the performance
			# Stopwords_list = get_stopwords_list('stopwords.txt')

			# Processing the negative file
			for sentence in neg:
				# Randomly assign the sentence into train data, development data or test data
				set_class = np.random.choice(['test', 'dev', 'train'], p=[.1, .1, .8])

				# Regular tokenize or use regular expression
				if tokenize:	
					tokens_list = self.simple_tokenize(sentence)
				else:
					tokens_list = self.re_tokenize(sentence)

				# Build the vocabulary using the words in training data
				tokenized_sentence = []
				for token in tokens_list:
					if token == '' or token == ' ':
						continue
					else:
						tokenized_sentence.append(token)
						if set_class == 'train':
							if token in vocab:
								vocab[token] += 1
							else:
								vocab[token] = 1

				# Class, y, text, length dictionary for single sentence
				cyt = {"y":0, "text": tokenized_sentence, "class": set_class, "len": len(tokenized_sentence)}
				info_list.append(cyt)

			# Same as the negative file
			for sentence in pos:
				set_class = np.random.choice(['test', 'dev', 'train'], p=[.1, .1, .8])

				if tokenize:
					tokens_list = self.simple_tokenize(sentence)
				else:
					tokens_list = self.re_tokenize(sentence)

				tokenized_sentence = []
				for token in tokens_list:
					if token == '' or token == ' ':
						continue
					else:
						tokenized_sentence.append(token)
						if set_class == 'train':
							if token in vocab:
								vocab[token] += 1
							else:
								vocab[token] = 1

				cyt = {"y":1, "text": tokenized_sentence, "class": set_class, "len": len(tokenized_sentence)}
				info_list.append(cyt)

			# Store the processed data in local
			pickle.dump(info_list, open('dataset.p', "wb"))
			pickle.dump(vocab, open('vocab.p', "wb"))


		return info_list, vocab

	''' 
	Get certain numbers of the data for testing the model(train data, development data, test data), default sampling size is 30000
	The ratio of train, dev, test data should be approximately 0.8 : 0.1 : 0.1
	'''
	def get_samples(self, info_list, vocab, sample_num = 30000):
		x_train = []
		y_train = []
		x_dev = []
		y_dev = []
		x_test = []
		y_test = []

		# Randomly select samples of `sampling size` from the collection of dicts
		# Each piece of infomation's format: {'y': 0 or 1, 'text': tokenized sentence, /
		# 'class': 'train' or 'dev' or 'test', 'len': length of the tokenized sentence}
		for piece in random.sample(info_list, int(sample_num)):
			if piece['class'] == 'train':
				x_train.append(piece['text'])
				if piece['y'] == 0:
					y_train.append(0)
				else:
					y_train.append(1)
			elif piece['class'] == 'test':
				x_dev.append(piece['text'])
				if piece['y'] == 0:
					y_dev.append(0)
				else:
					y_dev.append(1)
			else:
				x_test.append(piece['text'])
				if piece['y'] == 0:
					y_test.append(0)
				else:
					y_test.append(1)
		
		return [x_train, y_train, x_dev, y_dev, x_test, y_test]

	''' 
	Get a list of stopwords
	However, in this task it won't be use coz they were proved to be useless
	'''
	def get_stopwords_list(self, fname):
	    print("Getting stop words...")
	    f = open(fname, 'r', encoding='utf-8')
	    stopwords = f.read().split('\n')
	    return stopwords

	''' 
	Common way to tokenize the sentence and return a list of words
	'''
	def simple_tokenize(self, sentence):
		return jieba.cut(sentence)

	''' 
	Using regular expression to better tokenized the sentence 
	In this task I only exclude the alphabetical words
	'''
	def re_tokenize(self, sentence):
	    sentence = re.sub("[a-zA-Z]", "", sentence)
	    return jieba.cut(sentence) 

	''' 
	Read in all the text data for word2vec(*.txt) and connect them into a continuous, long sentence
	Prepare the full sentence for training the word2vec model
	'''
	def fullrec_for_word2vec(self, tokenize = True):
		neg = self.load_data('neg.txt')
		pos = self.load_data('pos.txt')
		if tokenize:
			all_sentence = []
			for sentence in neg:
			    tokens_list = jieba.cut(sentence)
			    tokenized_sentence = []
			    for token in tokens_list:
			        tokenized_sentence.append(token)
			    all_sentence.append(tokenized_sentence)
			for sentence in pos:	
			    tokens_list = jieba.cut(sentence)
			    tokenized_sentence = []
			    for token in tokens_list:
			        tokenized_sentence.append(token)
			    all_sentence.append(tokenized_sentence)
		else:
			pass
		return all_sentence

	''' 
	Load the Word2Vec model from local file
	Or create a new Word2Vec model with feeding text
	'''
	def get_word2vec(self, fname, embedding_size, override = False):
		if override:
			model = word2vec.Word2Vec.load(fname)
		else:	
			sentences = self.fullrec_for_word2vec()
			model = word2vec.Word2Vec(sentences, min_count=1, size=embedding_size)
			model.save(fname)
		return model

	''' 
	Load the word vectors from Word2Vec model
	Only words contained in the vocabulary will be considered
	'''
	def load_word_vectors(self, fname, vocab, embedding_size, override = False):
	    word_vecs = {}
	    model = self.get_word2vec(fname, embedding_size, override)
	    for word in vocab:
	        if word in model.wv:
	            word_vecs[word] = model.wv[word] 
	    return word_vecs

	''' 
	Get word matrix 
	W[i] is the vector for word indexed by i
	'''
	def get_word_embeddings(self, word_vecs, k=100):
	    vocab_size = len(word_vecs)
	    word_idx = defaultdict()
	    word_embeddings = np.zeros(shape=(vocab_size + 1, k), dtype='float32')            
	    i = 1
	    for word in word_vecs:
	        word_embeddings[i] = word_vecs[word]
	        word_idx[word] = i
	        i += 1
	    return word_embeddings, word_idx

	


class Logistic_Regression(object):
	""" Logistic Regression for sentiment prediction"""

	''' 
	Building a matrix for training and testing
	The shape of row_vec is (number of the sentences, vocabulary)
	row_vec[ii] iterate each of the sentence 
	row_vec[idx] indicates which words are contained in a sentence
	'''
	def get_X(self, x_raw, vocab):
		x_list = []
		vocab_list = list(vocab.keys())
		ii = 0
		row_vec = np.zeros(shape=(len(x_raw),len(vocab)))
		for row in x_raw:
			for word in row:
				if word in vocab_list:
					idx = vocab_list.index(word)
					# print('indexed done')
					row_vec[ii][idx] += 1
				else:
					continue
			if (ii % 5000) == 0:
				print('5000 row added')
			ii += 1
			# print(row_vec)
		return row_vec

	''' 
	Compute sigmoid value
	'''
	def sigmoid(self, x_row, bs):
	    return 1 / (1 + math.exp( -np.dot(x_row, bs.T)))

	''' 
	Compute the sum of log likelihood for the whole model
	We want this value to be as low as possible
	'''
	def log_likelihood(self, bs, X, y_train):
		ll = 0
		for i in range(len(y_train)):
			ll += (y_train[i] * np.dot(bs.T, X[i]) - math.log(1 + math.exp(np.dot(bs.T,X[i])))).sum()
		return ll

	''' 
	The derivative part of the model
	'''
	def compute_gradient(self, x_row, y, y_pred):
	    return ((y - y_pred) * x_row)

	''' 
	Initialize the weight vector to be array of zeros, the length is the size of the vocabulary
	Iterate the SGD process for num_step times
	Learning rate is a defined by experiment
	'''
	def logistic_regression(self, X, y_train, learning_rate=7e-5, num_step=3000000):
	    likelihood=[]
	    bs = np.zeros((len(X[0])), dtype='float64')
	    ii = 0
	    while ii < num_step:
	        for j in range(len(y_train)):
	            bs += self.compute_gradient(X[j], y_train[j], self.sigmoid(X[j], bs)) * learning_rate
	            ii += 1
	            start = time.time()
	            if (ii % 10000) == 0:
	                ll = self.log_likelihood(bs, X, y_train)
	                total = time.time() - start
	                print("Iteration: {}, likelihood: {}, time: {}".format(ii, ll, total))
	                likelihood.append(ll)

	    return bs,likelihood

	'''
	Figuring out the most important feature in classification
	'''
	def most_important_featrue(self, x_row, bs, vocab):
		vec = x_row * bs
		pos_index = vec.argmax()
		neg_index = vec.argmin()
		pos_term = vocab[pos_index]
		neg_term = vocab[neg_index]
		return pos_term, neg_term

	''' 
	Predict the sentiment of test data using the trained weight
	The result y_pred is an array of the predicted class of the test data
	'''
	def predict(self, bs, x_test, vocab, x_test_text):
	    vocab_list = list(vocab.keys())
	    y_pred = []
	    for i in range(x_test.shape[0]):
	        if i <=10:
	            print('The text of the original review: {}'.format(' '.join(x_test_text[i])))
	        if self.sigmoid(x_test[i], bs) >= 0.5:
	            if i <=10:
	                pos_term, neg_term = self.most_important_featrue(x_test[i], bs, vocab_list)
	                print('------Most important feature in this POSITIVE(predicted) review is: {} \n'.format(pos_term))
	            # print('------Most important feature in this review is: {}'.format(neg_term))
	            y_pred.append(1.0)
	        else:
	            if i <=10:
	                pos_term, neg_term = self.most_important_featrue(x_test[i], bs, vocab_list)
	                print('------Most important feature in this NEGATIVE(predicted) review is: {} \n'.format(neg_term))
	            # print('------Most important feature in this review is: {}'.format(pos_term))
	            y_pred.append(0.0)
	    # print(y_pred)
	    return np.array(y_pred, dtype='float32')

class CNN(object):
	""" Convolutional Neural Network for sentiment prediction"""

	''' 
	Indexed the sentence and pad them with zeros into same length
	The longest sentence is no longer than 200 tokens
	The word_idx is a map for identifying the position of a word in the word2vec matrix
	'''
	def indexed_sentence(self, sent, word_idx, max_length = 150, kernel_size = 5):
		padded_sent = []
		padding_size = kernel_size - 1
		# print(word_idx.keys())
		for i in range(padding_size):
			padded_sent.append(0)
		for token in sent:
			if token in word_idx:
				padded_sent.append(word_idx[token])
		while len(padded_sent) < max_length + 2 * padding_size:
			padded_sent.append(0)
		return padded_sent

	'''
	Prepare the train, dev, test data for cnn model
	'''
	def get_indexed_data(self, dataset, word_idx, max_length = 150, kernel_size = 5):
		x_train = []
		y_train = []
		x_dev = []
		y_dev = []
		x_test = []
		y_test = []

		for xtrain in dataset[0]:
			x_train.append(self.indexed_sentence(xtrain, word_idx))
		for ytrain in dataset[1]: 
			y_train.append(int(ytrain))
		for xdev in dataset[2]:
			x_dev.append(self.indexed_sentence(xdev, word_idx))
		for ydev in dataset[3]: 
			y_dev.append(int(ydev))
		for xtest in dataset[4]:
			x_test.append(self.indexed_sentence(xtest, word_idx))
		for ytest in dataset[5]: 
			y_test.append(int(ytest))

		# Eg. x_train.shape = (16000, 208)
		x_train = np.array(x_train)
		# Eg. y_train.shape = (16000,)
		y_train = np.array(y_train)
		x_dev = np.array(x_dev)
		y_dev = np.array(y_dev)
		x_test = np.array(x_test)
		y_test = np.array(y_test)

		print ('x_train.shape = {}'.format(x_train.shape))
		print ('y_train.shape = {}'.format(y_train.shape))
		print ('x_dev.shape = {}'.format(x_dev.shape))
		print ('y_dev.shape = {}'.format(y_dev.shape))
		print ('x_test.shape = {}'.format(x_test.shape))
		print ('y_test.shape = {}'.format(y_test.shape))

		return [x_train, y_train, x_dev, y_dev, x_test, y_test]

	'''
	Process each class of data
	After processing, an additionall row is added to y data, if y[0, ii] = 0 then the y[1, ii] = 1 
	'''
	def map_idx_data_to_embedding(self, x_data, y_data, W):
		num_sent = x_data.shape[0]
		conv_input_height = x_data.shape[1]

		x_ult = np.zeros(shape=(num_sent, conv_input_height))
		y_ult = np.zeros(shape=(num_sent, 2))

		for ii in range(0, num_sent-1):
			for jj in range(0, conv_input_height-1):
				x_ult[ii][jj] = x_data[ii][jj]
			y_ult[ii][y_data[ii]] = 1

		# Eg. x_ult = array([[0, 0, 0, 0 ... , 0, 0, 0, 0], [0, 0, 0, 0 ... , 0, 0, 0, 0] ..., [0, 0, 0, 0 ... , 0, 0, 0, 0]])
		# Eg. y_ult = array([[0, 1, 0, 1 ... , 0, 0, 1, 0], [1, 0, 1, 0 ... , 1, 1, 0, 1]])
		return x_ult, y_ult

	'''
	Constructing the CNN model
	Based on Yoon Kim's paper
	Most of this part comes from : https://github.com/vsl9/Sentiment-Analysis-with-Convolutional-Networks/blob/master/Train%20CNN%20IMDB.ipynb
	'''
	def building_CNN(self, conv_input_width, conv_input_height, W):
		backend.set_image_dim_ordering('th')

		# Number of feature maps (outputs of convolutional layer)
		conv_layer_num = 300

		# kernel size of convolutional layer
		kernel_size = 8

		model = Sequential()

		# Embedding layer (lookup table of trainable word vectors)
		model.add(Embedding(input_dim=W.shape[0], 
		                    output_dim=W.shape[1], 
		                    input_length=conv_input_height,
		                    weights=[W], 
		                    W_constraint=unitnorm()))

		# Reshape word vectors from Embedding to tensor format suitable for Convolutional layer
		model.add(Reshape((1, conv_input_height, conv_input_width)))

		# first convolutional layer
		model.add(Convolution2D(conv_layer_num, 
			                        kernel_size, 
			                        conv_input_width, 
			                        border_mode='valid', 
			                        W_regularizer=l2(0.0001)))

		# ReLU activation
		model.add(Activation('relu'))

		# aggregate data in every feature map to scalar using MAX operation
		model.add(MaxPooling2D(pool_size=(conv_input_height-kernel_size+1, 1)))

		model.add(Flatten())
		model.add(Dropout(0.5))

		# Inner Product layer (as in regular neural network, but without non-linear activation function)
		model.add(Dense(2))

		# SoftMax activation; actually, Dense+SoftMax works as Multinomial Logistic Regression
		model.add(Activation('softmax'))

		# Custom optimizers could be used, though right now standard adadelta is employed
		opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-6)
		model.compile(loss='categorical_crossentropy', 
		              optimizer=opt,
		              metrics=['accuracy'])

		return model




if __name__=="__main__":

	argparser = argparse.ArgumentParser()
	argparser.add_argument("--task", help = "Which task am I working on: 1). 'dp': Data Processing, 2). 'lg': Logistic Regression, 3). 'cnn': CNN",
	                       type = str, default = "dp", required = True)
	argparser.add_argument("--neg_fname", help = "Indicate the negative file for dataset",
	                       type = str, default = "neg.txt", required = False)
	argparser.add_argument("--pos_fname", help = "Indicate the positive file for dataset",
	                       type = str, default = "pos.txt", required = False)
	argparser.add_argument("--override", help = "Load local files or create new files",
	                       type = bool, default = False, required = False)
	argparser.add_argument("--sample_num", help = "How many data am I using to run the model(s)",
	                       type = int, default = 30000, required = False)
	argparser.add_argument("--embedding_size", help = "What's the vector size of the Word2Vec embeddings for each word",
	                       type = int, default = 100, required = False)
	argparser.add_argument("--learning_rate", help = "How many epochs am I iterating in CNN training process",
	                       type = float, default = 6e-5, required = False)
	argparser.add_argument("--lg_iteration_num", help = "How many epochs am I iterating in CNN training process",
	                       type = int, default = 3000000, required = False)
	argparser.add_argument("--epoch_cnn", help = "How many epochs am I iterating in CNN training process",
	                       type = int, default = 3, required = False)
	args = argparser.parse_args()

	#.................................................................................
	#... Data Processing
	#.................................................................................
	if args.task == 'dp':
		dp = Data_Processing()
		# Load or Create dataset
		print("Loading data ......")
		info_list, vocab = dp.get_data_and_labels(args.neg_fname, args.pos_fname, args.override)
		print("Dataset created or retrived!")
		print('Numbers of sentences we have proccessed: {}'.format(len(info_list)))
		print('Size of the vocabulary: {}'.format(len(vocab)))

		# Load or Create word2vec matrix
		print("Loading word2vec ......")
		word_vecs = dp.load_word_vectors('simple_word2vec.model', vocab, args.embedding_size, args.override)
		W, word_idx = dp.get_word_embeddings(word_vecs, args.embedding_size)
		print("Word2vec embeddings created or retrived!")

	#.................................................................................
	#... Logistic Regression for sentiment prediction
	#.................................................................................
	elif args.task == 'lg':
		dp = Data_Processing()

		# Load or Create dataset
		print("Loading data ......")
		info_list, vocab = dp.get_data_and_labels(args.neg_fname, args.pos_fname, args.override)
		print("Dataset created or retrived!")

		# Extract samples from the dataset
		# tdt_dataset = [x_train, y_train, x_dev, y_dev, x_test, y_test]
		# tdt_dataset contains raw data of each class
		tdt_dataset = dp.get_samples(info_list, vocab, args.sample_num)

		lg = Logistic_Regression()

		# Constructing matrix for x_train and x_test
		# Columns are vocabularys and rows are each sentence
		# Sparse matrix, each cell keep tracks of the num of times a token occurs in the sentence
		print("building x matrix for training ...")
		x_train = lg.get_X(tdt_dataset[0], vocab)
		print("building x matrix for testing ...")    
		x_test = lg.get_X(tdt_dataset[4], vocab)

		# Constructing matrix for y_train and y_test
		# y = [0, 0, 1, 0 .... 0, 1, 1, 0, 1]
		y_train = np.array(tdt_dataset[1])
		y_test = np.array(tdt_dataset[5], dtype='float32')

		# Performing logistic regression
		print("performing logistic regression ...")    
		bs, likelihoods = lg.logistic_regression(x_train, y_train, args.learning_rate, args.lg_iteration_num)

		# Predict the x_test using the weight trained from the model
		y_pred = lg.predict(bs, x_test, vocab, tdt_dataset[4])

		# F1-score to inspect the performance
		a = f1_score(y_test, y_pred)
		print("The Logistic Regression model achieve a F1-Score of {}: ".format(a))

	#.................................................................................
	#... CNN for sentiment prediction
	#.................................................................................
	elif args.task == 'cnn':
		dp = Data_Processing()

		# Load or Create dataset
		print("Loading data ......")
		info_list, vocab = dp.get_data_and_labels(args.neg_fname, args.pos_fname, args.override)
		print("Dataset created or retrived!")
		print('Numbers of sentences we have proccessed: {}'.format(len(info_list)))
		print('Size of the vocabulary: {}'.format(len(vocab)))

		# Load or Create word2vec matrix
		print("Loading word2vec ......")
		word_vecs = dp.load_word_vectors('simple_word2vec.model', vocab, args.embedding_size, args.override)
		W, word_idx = dp.get_word_embeddings(word_vecs, args.embedding_size)
		print("Word2vec embeddings created or retrived!")

		cnn = CNN()

		# Extract samples from the dataset
		# tdt_dataset = [x_train, y_train, x_dev, y_dev, x_test, y_test]
		# tdt_dataset contains raw data of each class
		tdt_dataset = dp.get_samples(info_list, vocab, args.sample_num)

		# Process the raw data into cnn_dataset
		# Each sentence is padded to same length
		# Words in the sentence are represented by index (for looking up in the word2vec matrix)
		cnn_dataset = cnn.get_indexed_data(tdt_dataset, word_idx)

		# Train, dev, test data
		x_train, y_train = cnn.map_idx_data_to_embedding(cnn_dataset[0], cnn_dataset[1], W)
		x_dev, y_dev = cnn.map_idx_data_to_embedding(cnn_dataset[2], cnn_dataset[3], W)
		x_test, y_test = cnn.map_idx_data_to_embedding(cnn_dataset[4], cnn_dataset[5], W)

		conv_input_width = W.shape[1]
		conv_input_height = x_train.shape[1]

		# Construct the CNN model
		model = cnn.building_CNN(conv_input_width, conv_input_height, W)

		# Train the CNN model
		# Validate the model on dev data
		epoch = 0
		N_epoch = args.epoch_cnn
		dev_acc = []

		for i in range(N_epoch):
			model.fit(x_train, y_train, batch_size=50, nb_epoch=1, verbose=1)
			output = model.predict_proba(x_dev, batch_size=10, verbose=1)
			# find validation accuracy using the best threshold value t
			accuracy = []
			ts = []
			for t in np.arange(0.0, 1.0, 0.01):
				accuracy.append(np.sum((output[:,1]>t)==(y_dev[:,1]>0.5))*1.0/len(output))
				ts.append(t)
			acc = np.max(accuracy)
			threshold = ts[accuracy.index(acc)]

			# acc = np.max([np.sum((output[:,1]>t)==(y_dev[:,1]>0.5))*1.0/len(output) for t in np.arange(0.0, 1.0, 0.01)])
			dev_acc.append(acc)
			print ('Epoch {}: validation accuracy = {:.3%}'.format(epoch, acc))
			epoch += 1
		print(threshold)

		# Use the CNN model to predict the sentiment
		output2 = model.predict_proba(x_test, batch_size=10, verbose=1)
		y_pred = []
		neg_pred = 0
		pos_pred = 0
		for each in output2:
			if each[0] > 0.5:
				neg_pred += 1
				y_pred.append(0)
			else:
				pos_pred += 1
				y_pred.append(1)

		# F1-score to inspect the performance
		neg_num = 0
		pos_num = 0
		right = 0
		for ii, yy in enumerate(cnn_dataset[5]):
			if yy == y_pred[ii]:
				right += 1
			if yy == 0:
				neg_num += 1
			else:
				pos_num += 1

		print(right)

		print("Predicted numbers of NEGATIVE reviews: {}".format(neg_pred))
		print("Predicted numbers of POSITIVE reviews: {} \n".format(pos_pred))
		print("Real numbers of NEGATIVE reviews: {}".format(neg_num))
		print("Real numbers of POSITIVE reviews: {} \n".format(pos_num))

		a = f1_score(cnn_dataset[5], y_pred)
		print("The CNN model achieve a F1-Score of {}: ".format(a))