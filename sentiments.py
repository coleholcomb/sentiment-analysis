#%%
import numpy as np
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
        self.class_priors = np.zeros(self.nclasses) + self.nclasses
        self.likelihood = np.zeros((self.nclasses, self.nfeatures)) + 1

        for i in range(self.nexamples):
            y = int(self.labels[i])
            self.class_priors[y] += 1
            for j in range(self.nfeatures):
                if self.features[i,j] == 1:
                    self.likelihood[y,j] += 1 
        self.class_priors = self.class_priors/self.nexamples
        self.likelihood = self.likelihood/self.nexamples

        print('N_features = {}'.format(self.nfeatures))
        print('N_classes  = {}'.format(self.nclasses))
        print('N_examples = {}'.format(self.nexamples))

        print(self.class_priors)
        print(self.likelihood)
    
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

features=pd.read_csv('out_bag_of_words_5.csv', sep=',',header=None)
labels = np.genfromtxt('out_classes_5.txt', delimiter='\n')

testfeatures = pd.read_csv('test_bag_of_words_0.csv', sep=',', header=None)
testlabels = np.genfromtxt('test_classes_0.txt', delimiter='\n')



#%%
nb = NaiveBayes()
nb.train(features,labels)
nb.test(testfeatures, testlabels)
