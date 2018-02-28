----------------------------------------------
COS 424: Sentiment Analysis Task (Spring 2018)
----------------------------------------------

This directory comprises the following files

- train.txt, a training set of 2400 sentiment labelled sentences 
(from Amazon, Yelp and iMDB) in the following form:
<sample num>	<sentence> 	<sentiment> 
where <sentiment> is either 0 (negative) or 1 (positive).

- test.txt, a test set of 600 labelled sentences, in the same format.

- preprocessSentences.py, called as follows: 
	python ./preprocessSentences.py -p <data> -o <out> -v <vocab> -d <datafile> --threshold <threshold>
<data> is the path of the directory containing train.txt,
<out> is an optional argument specifying the prefix of output files, and 
<vocab> is an optional argument specifying the path to an existing 
vocabulary file. 
<datafile> is an optional argument specifying the dataset file name (default is 'train.txt')
<threshold> is an optional argument specifying the threshold for including words in the vocabulary (default is 1)

The script generates four output files in <data>: 
	- A vocabulary file 'out_vocab_*.txt’ (if one is not specified 
when calling the script) with tokenized words from the training data 
(where '*' is the word count threshold, set to default as 1 in the 
script), 
	- A list of training samples numbers in 
'out_samples_classes_*.txt',
	- A list of sentiment labels corresponding to each training 
sample, 'out_classes_*.txt', and
	- A 'bag of words' featurized representation of each sentence in 
the training set, 'out_bag_of_words_*.csv'

- knn-classify.py, an implementation of the KNN classifier

- sentiments.py, a script containing classes related to Naive Bayes Classification, and some feature filtering
				 techniques (Mutual Information and Maximal Discriminative Information/J-Divergence)

- sentiments-nb.ipynb, a jupyter notebook for doing all data processing, analysis, and figure production 
					   for this assignment (excluding KNN classification and NB ROC curves)

----------------------------------------------
