#%%
class NaiveBayes():
    '''

    '''
    def __init__(self):
        self.nclasses = 0
        self.nfeatures = 0
    
    def train(self, features, labels):
        self.features = features.values
        self.labels = labels

        self.nclasses = np.unique(labels).shape[0]
        self.nexamples = features.shape[0]
        self.nfeatures = features.shape[1]
        self.class_priors = np.zeros(self.nclasses)
        self.log_likelihood = np.zeros((self.nclasses, self.nfeatures))

        for i in range(self.nexamples):
            y = int(self.labels[i])
            self.class_priors[y] += 1
            for j in range(self.nfeatures):
                if self.features[i,j] == 1:
                    self.log_likelihood[y,j] += 1 
        self.class_priors = self.class_priors/self.nexamples
        self.log_likelihood = self.log_likelihood/self.nexamples

        print('N_features = {}'.format(self.nfeatures))
        print('N_classes  = {}'.format(self.nclasses))
        print('N_examples = {}'.format(self.nexamples))

        print(self.class_priors)
        print(self.log_likelihood)
    
    def test(self, features, labels):
        


#%%
import pandas as pd
import numpy as np
features=pd.read_csv('out_bag_of_words_5.csv', sep=',',header=None)
labels = np.genfromtxt('out_classes_5.txt', delimiter='\n')

np.unique(labels).shape[0]

#%%
nb = NaiveBayes()
nb.train(features,labels)
