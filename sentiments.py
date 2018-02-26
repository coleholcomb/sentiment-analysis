'''
Notebook for implementing binary classification methods on product reviews.
'''
#%%
from abc import ABCMeta, abstractmethod
from operator import itemgetter

class FeatureFilter(metaclass=ABCMeta):
    '''
    Abstract base class for performing feature filtering for binary classification.
    '''
    @abstractmethod
    def __init__(self, features, labels):
        '''
        Given a bag-of-words feature representation and associated class labels, 
        calculate term probability, class probability, and joint probability

        inputs:
            features:   bag-of-words training matrix
            labels:     class label vector
        
        outputs:
            No output
        '''
        self.features = features.values
        self.labels = labels

        uni, cnts = np.unique(self.labels, return_counts=True)
        self.nclasses = uni.shape[0]
        self.nclass = np.array(cnts)
        self.nexamples = features.shape[0]
        self.nfeatures = features.shape[1]

        self.terms_prob = np.zeros(self.nfeatures) + 1
        self.joint_prob = np.zeros((self.nclasses, self.nfeatures)) + 1/self.nclasses

        for i in range(self.nexamples):
            y = int(self.labels[i])
            for j in range(self.nfeatures):
                if self.features[i,j] >= 1:
                    self.terms_prob[j] += 1
                    self.joint_prob[y,j] += 1
    
        self.class_prob = self.nclass/self.nexamples
        self.terms_prob = self.terms_prob/(self.nexamples + 1*self.nfeatures)
        self.joint_prob = self.joint_prob/(self.nexamples + 1*self.nfeatures)

        self.terms_nprob = 1 - self.terms_prob
        self.class_nprob = 1 - self.class_prob

        self.joint_ntprob = np.outer(self.class_prob, np.ones(self.nfeatures)) - self.joint_prob
        self.joint_ncprob = self.terms_prob - self.joint_prob
        self.joint_nprob = self.terms_nprob - self.joint_ntprob


class MutualInformation(FeatureFilter):
    '''
    Calculate the Mutual Information for a given set of vocabulary and classes.
    '''
    def __init__(self, features, labels, vocab, n):
        '''
        Given a bag-of-words feature representation and associated class labels, 
        calculate term probability, class probability, joint probability, and
        Mutual Information for each term.

        input from preprocessSentences.py:
            features:   bag-of-words training matrix
            labels:     class label vector
            vocab:      vocab list
            n:          number of features to reduce to

        save to file:
            reduced versions of inputs
        '''
        super().__init__(features, labels)
        self.mi  =( self.joint_prob*np.log2(self.joint_prob
                                           /np.outer(self.class_prob, self.terms_prob))
                  + self.joint_ntprob*np.log2(self.joint_ntprob
                                           /np.outer(self.class_prob, self.terms_nprob))
                  + self.joint_ncprob*np.log2(self.joint_ncprob
                                           /np.outer(self.class_nprob, self.terms_prob))
                  + self.joint_nprob*np.log2(self.joint_nprob
                                           /np.outer(self.class_nprob, self.terms_nprob)) )[0]

        self.vocab = sorted([(x,y,z) for x,y,z in zip(range(self.nfeatures), vocab, self.mi)], 
                            key=itemgetter(2), reverse=True)[:n]
        
        inds = np.array(self.vocab).T[0].astype(int)

        #save ordered vocabulary list of n most informative words, with indicies and MI
        with open('mi_'+str(n)+'_vocab_1.txt', 'w', encoding='utf-8') as f:
            for l in self.vocab:
                f.write(str(l[1])+'\n')

        with open('mi_'+str(n)+'_bag_of_words_1.csv', 'w', encoding='utf-8') as f:
            for i in range(self.nexamples):
                for j in range(n):
                    k = inds[j]
                    f.write(str(self.features[i,k]))
                    if j < n-1:
                        f.write(',')
                    else:
                        f.write('\n')


class MaxDisc(FeatureFilter):
    '''
    Calculate the Maximally Discriminative features for a given set of vocabulary and classes.
    
    ref: https://arxiv.org/pdf/1602.02850.pdf
    '''
    def __init__(self, features, labels, vocab, n):
        '''
        Given a bag-of-words feature representation and associated class labels, 
        calculate term probability, class probability, joint probability, and
        J-Divergence for each term.

        input from preprocessSentences.py:
            features:   bag-of-words training matrix
            labels:     class label vector
            vocab:      vocab list
            n:          number of features to reduce to

        save to file:
            reduced versions of inputs
        '''
        super().__init__(features, labels)
        self.kl01  = self.joint_prob[0]*np.log2(self.joint_prob[0]/self.joint_prob[1])
        self.kl10  = self.joint_prob[1]*np.log2(self.joint_prob[1]/self.joint_prob[0])
        self.jd = self.kl01 + self.kl10

        self.vocab = sorted([(x,y,z) for x,y,z in zip(range(self.nfeatures), vocab, self.jd)], 
                            key=itemgetter(2), reverse=True)[:n]

        self.vocabkl01 = sorted([(x,y,z) for x,y,z in zip(range(self.nfeatures), vocab, self.kl01)], 
                            key=itemgetter(2), reverse=True)[:n]
        self.vocabkl10 = sorted([(x,y,z) for x,y,z in zip(range(self.nfeatures), vocab, self.kl10)], 
                            key=itemgetter(2), reverse=True)[:n]
        
        inds = np.array(self.vocab).T[0].astype(int)

        #save ordered vocabulary list of n most informative words, with indicies and MI
        with open('jd_'+str(n)+'_vocab_1.txt', 'w', encoding='utf-8') as f:
            for l in self.vocab:
                f.write(str(l[1])+'\n')

        with open('jd_'+str(n)+'_bag_of_words_1.csv', 'w', encoding='utf-8') as f:
            for i in range(self.nexamples):
                for j in range(n):
                    k = inds[j]
                    f.write(str(self.features[i,k]))
                    if j < n-1:
                        f.write(',')
                    else:
                        f.write('\n')
#%%
vocab = open('out_vocab_1.txt', 'r', encoding='utf-8').read(-1).split('\n')
#if '$' in vocab[0]:
#    vocab[0] = '$'
filt = MutualInformation(features, labels, vocab, 20)
for i in range(20):
    print(filt.vocab[i])


#%%
inds = np.array(filt.vocab).T[0].astype(int)
inds[0]

#%%
filt.joint_prob[0]

#%%
import matplotlib.pyplot as plt

plt.hist(filt.mi[0]+filt.mi[1], bins=200)

#%%
import numpy as np
class NaiveBayes():
    '''
    Naive Bayes classifier for sentiment analysis on feature output from
    preprocessSentences.py
    '''
    def __init__(self):
        self.nclasses = 0
        self.nfeatures = 0
    
    def train(self, features, labels):
        self.features = features.values
        self.labels = labels

        uni, cnts = np.unique(labels, return_counts=True)
        self.nclasses = uni.shape[0]
        self.nclass = cnts
        self.nexamples = features.shape[0]
        self.nfeatures = features.shape[1]
        self.class_priors = np.zeros(self.nclasses)
        self.likelihood = np.zeros((self.nclasses, self.nfeatures)) + 1

        for i in range(self.nexamples):
            y = int(self.labels[i])
            self.class_priors[y] += 1
            for j in range(self.nfeatures):
                if self.features[i,j] == 1:
                    self.likelihood[y,j] += 1 
        self.class_priors = self.class_priors/self.nexamples
        self.likelihood = self.likelihood/(self.nexamples + self.nfeatures)

        print('N_features = {}'.format(self.nfeatures))
        print('N_classes  = {}'.format(self.nclasses))
        print('N_examples = {}'.format(self.nexamples))
    
    def test(self, features, labels):
        features = features.values
        ntest = labels.shape[0]
        pc = np.empty((ntest, self.nclasses))
        
        for i in range(ntest):
            for c in range(self.nclasses):
                pc[i, c] = np.log(self.class_priors[c])
                for j in range(self.nfeatures):
                    if features[i,j] == 1:
                        pc[i,c] += np.log(self.likelihood[c,j])
                    else:
                        pc[i,c] += np.log(1 - self.likelihood[c,j])
        
        pc[i,c] = np.exp(pc[i,c])
        prediction = np.argmax(pc, axis=-1)
        
        correct, false, fp, fn = 0, 0, 0, 0
        for i in range(ntest):
            if prediction[i] == labels[i]:
                correct += 1
            elif prediction[i] == 1:
                fp += 1
                false += 1
            else:
                fn +=1
                false +=1

        print('Percentage correct:         {}'.format(correct/ntest))
        print('Percentage incorrect:       {}'.format(false/ntest))
        print('Percentage false positives: {}'.format(fp/ntest))
        print('Percentage false negatives: {}'.format(fn/ntest))

#%%
import pandas as pd
import numpy as np

jd_thresholds = [10,50,100,200,300,400,500,1621]
mi_thresholds = [10,50,100,200,300,400,500,1621]
thresholds = [1]

for i in jd_thresholds:
    print('jd: '+str(i))
    testfeatures = pd.read_csv('jd_'+str(i)+'_test_bag_of_words_0.csv', sep=',', header=None)
    testlabels = np.genfromtxt('jd_'+str(i)+'_test_classes_0.txt', delimiter='\n')

    features=pd.read_csv('jd_'+str(i)+'_bag_of_words_1.csv', sep=',',header=None)
    labels = np.genfromtxt('out_classes_1.txt', delimiter='\n')

    nb = NaiveBayes()
    nb.train(features,labels)
    nb.test(testfeatures, testlabels)
    print('\n')

for i in mi_thresholds:
    print('mi: '+str(i))
    testfeatures = pd.read_csv('mi_'+str(i)+'_test_bag_of_words_0.csv', sep=',', header=None)
    testlabels = np.genfromtxt('mi_'+str(i)+'_test_classes_0.txt', delimiter='\n')

    features=pd.read_csv('mi_'+str(i)+'_bag_of_words_1.csv', sep=',',header=None)
    labels = np.genfromtxt('out_classes_1.txt', delimiter='\n')

    nb = NaiveBayes()
    nb.train(features,labels)
    nb.test(testfeatures, testlabels)
    print('\n')

for i in thresholds:
    print('all: '+str(i))
    testfeatures = pd.read_csv('test'+str(i)+'_bag_of_words_0.csv', sep=',', header=None)
    testlabels = np.genfromtxt('test'+str(i)+'_classes_0.txt', delimiter='\n')

    features=pd.read_csv('out_bag_of_words_'+str(i)+'.csv', sep=',',header=None)
    labels = np.genfromtxt('out_classes_'+str(i)+'.txt', delimiter='\n')

    nb = NaiveBayes()
    nb.train(features,labels)
    nb.test(testfeatures, testlabels)
    print('\n')

#%%
from sklearn.svm import SVC, LinearSVC

mi_thresholds = [10,50,100,200,300,400,500,1621]
thresholds = [1]

for i in mi_thresholds:
    print('mi: '+str(i))
    testfeatures = pd.read_csv('mi_'+str(i)+'_test_bag_of_words_0.csv', sep=',', header=None)
    testlabels = np.genfromtxt('mi_'+str(i)+'_test_classes_0.txt', delimiter='\n')

    features=pd.read_csv('mi_'+str(i)+'_bag_of_words_1.csv', sep=',',header=None)
    labels = np.genfromtxt('out_classes_1.txt', delimiter='\n')

    svc = SVC()
    svc.fit(features, labels)

    linsvc = LinearSVC()
    linsvc.fit(features, labels) 
    testpred = svc.predict(testfeatures)
    ntest = testlabels.shape[0]
    correct, false, fp, fn = 0, 0, 0, 0
    for i in range(ntest):
        if testpred[i] == testlabels[i]:
            correct += 1
        elif testpred[i] == 1:
            fp += 1
            false += 1
        else:
            fn +=1
            false +=1

    print('Percentage correct:         {}'.format(correct/ntest))
    print('Percentage incorrect:       {}'.format(false/ntest))
    print('Percentage false positives: {}'.format(fp/ntest))
    print('Percentage false negatives: {}'.format(fn/ntest))

    testpred = linsvc.predict(testfeatures)
    ntest = testlabels.shape[0]
    correct, false, fp, fn = 0, 0, 0, 0
    for i in range(ntest):
        if testpred[i] == testlabels[i]:
            correct += 1
        elif testpred[i] == 1:
            fp += 1
            false += 1
        else:
            fn +=1
            false +=1

    print('Percentage correct:         {}'.format(correct/ntest))
    print('Percentage incorrect:       {}'.format(false/ntest))
    print('Percentage false positives: {}'.format(fp/ntest))
    print('Percentage false negatives: {}'.format(fn/ntest))

    print('\n')

for i in thresholds:
    print('all: '+str(i))
    testfeatures = pd.read_csv('test'+str(i)+'_bag_of_words_0.csv', sep=',', header=None)
    testlabels = np.genfromtxt('test'+str(i)+'_classes_0.txt', delimiter='\n')

    features=pd.read_csv('out_bag_of_words_'+str(i)+'.csv', sep=',',header=None)
    labels = np.genfromtxt('out_classes_'+str(i)+'.txt', delimiter='\n')

    svc = SVC()
    svc.fit(features, labels)

    linsvc = LinearSVC()
    linsvc.fit(features, labels) 
    testpred = svc.predict(testfeatures)
    ntest = testlabels.shape[0]
    correct, false, fp, fn = 0, 0, 0, 0
    for i in range(ntest):
        if testpred[i] == testlabels[i]:
            correct += 1
        elif testpred[i] == 1:
            fp += 1
            false += 1
        else:
            fn +=1
            false +=1

    print('Percentage correct:         {}'.format(correct/ntest))
    print('Percentage incorrect:       {}'.format(false/ntest))
    print('Percentage false positives: {}'.format(fp/ntest))
    print('Percentage false negatives: {}'.format(fn/ntest))

    testpred = linsvc.predict(testfeatures)
    ntest = testlabels.shape[0]
    correct, false, fp, fn = 0, 0, 0, 0
    for i in range(ntest):
        if testpred[i] == testlabels[i]:
            correct += 1
        elif testpred[i] == 1:
            fp += 1
            false += 1
        else:
            fn +=1
            false +=1

    print('Percentage correct:         {}'.format(correct/ntest))
    print('Percentage incorrect:       {}'.format(false/ntest))
    print('Percentage false positives: {}'.format(fp/ntest))
    print('Percentage false negatives: {}'.format(fn/ntest))

    print('\n')



#%%
testpred = svc.predict(testfeatures)
ntest = testlabels.shape[0]
correct, false, fp, fn = 0, 0, 0, 0
for i in range(ntest):
    if testpred[i] == testlabels[i]:
        correct += 1
    elif testpred[i] == 1:
        fp += 1
        false += 1
    else:
        fn +=1
        false +=1

print('Percentage correct:         {}'.format(correct/ntest))
print('Percentage incorrect:       {}'.format(false/ntest))
print('Percentage false positives: {}'.format(fp/ntest))
print('Percentage false negatives: {}'.format(fn/ntest))

testpred = linsvc.predict(testfeatures)
ntest = testlabels.shape[0]
correct, false, fp, fn = 0, 0, 0, 0
for i in range(ntest):
    if testpred[i] == testlabels[i]:
        correct += 1
    elif testpred[i] == 1:
        fp += 1
        false += 1
    else:
        fn +=1
        false +=1

print('Percentage correct:         {}'.format(correct/ntest))
print('Percentage incorrect:       {}'.format(false/ntest))
print('Percentage false positives: {}'.format(fp/ntest))
print('Percentage false negatives: {}'.format(fn/ntest))

#%%
from sklearn.tree import DecisionTreeClassifier

dtgini = DecisionTreeClassifier()
dtgini.fit(features, labels)

dtentropy = DecisionTreeClassifier(criterion='entropy')
dtentropy.fit(features, labels) 

#%%
testpred = dtgini.predict(testfeatures)
ntest = testlabels.shape[0]
correct, false, fp, fn = 0, 0, 0, 0
for i in range(ntest):
    if testpred[i] == testlabels[i]:
        correct += 1
    elif testpred[i] == 1:
        fp += 1
        false += 1
    else:
        fn +=1
        false +=1

print('Percentage correct:         {}'.format(correct/ntest))
print('Percentage incorrect:       {}'.format(false/ntest))
print('Percentage false positives: {}'.format(fp/ntest))
print('Percentage false negatives: {}'.format(fn/ntest))

testpred = dtentropy.predict(testfeatures)
ntest = testlabels.shape[0]
correct, false, fp, fn = 0, 0, 0, 0
for i in range(ntest):
    if testpred[i] == testlabels[i]:
        correct += 1
    elif testpred[i] == 1:
        fp += 1
        false += 1
    else:
        fn +=1
        false +=1

print('Percentage correct:         {}'.format(correct/ntest))
print('Percentage incorrect:       {}'.format(false/ntest))
print('Percentage false positives: {}'.format(fp/ntest))
print('Percentage false negatives: {}'.format(fn/ntest))

#%%
from sklearn.feature_selection import mutual_info_classif

mi = mutual_info_classif(features, labels)
print(mi)
#%%
mi.shape