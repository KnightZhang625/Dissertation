import re
import os
import sys
import pickle
import numpy as np
from nltk import download
from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from sklearn import cluster
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class Classify(object):

    def __init__(self,model_type='tf_idf',n=3,train=True,type='train'):
        download('punkt')
        download('stopwords')
        stop_words = stopwords.words('english')
        self.type = type

        if self.type == 'train':
            train_file = open('cache/train_tgt_inform_def_trim=false_full_delex=true.txt','r')
            self.file = train_file.read()
            train_file.close()
            self.sentences = self.file.split('\n')[:-1]

            src_file = open('cache/train_src_inform_def_trim=false_full_delex=true.txt','r')
            self.src = src_file.read()
            src_file.close()
            self.raw_sentence = self.src.split('\n')[:-1]
        else:
            train_file = open('cache/valid_tgt_inform_def_trim=false_full_delex=true.txt','r')
            self.file = train_file.read()
            train_file.close()
            self.sentences = self.file.split('\n')[:-1]

            src_file = open('cache/valid_src_inform_def_trim=false_full_delex=true.txt','r')
            self.src = src_file.read()
            src_file.close()
            self.raw_sentence = self.src.split('\n')[:-1]

        self.trainData = []
        self.temp = {}
        self.n = n

        if model_type == 'tf_idf' and train == True:
            print('tf-idf training ...')
            self._preProcess()
            self._loadModel()
            self.result = np.zeros([])
        elif model_type == 'word2vec' and train == True:
            print('word2vec training ...')
            self._preProcess_2()
            self._loadModel_2()
        else:
            print('call by other file ...')

    # tf-idf
    def _preProcess(self):
        for sentence in self.sentences:
            sentence_off_punc = re.sub('[,.]+','',sentence)
            self.trainData.append(sentence_off_punc)

    def _loadModel(self):
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.trainData)

    def calculate_distance(self):
        print(self.tfidf_matrix.shape)
        return cosine_similarity(self.tfidf_matrix[1], self.tfidf_matrix[120:125])

    def classify(self):
        self.k_means = cluster.KMeans(n_clusters=self.n)
        self.k_means.fit(self.tfidf_matrix)
        self.result = self.k_means.labels_
        self.save_data()

    # word embedding
    def _preProcess_2(self):
        for sentence in self.sentences:
            temp = [w for w in sentence.split()]
            
            for index,word in enumerate(temp):
                if '@' in word:
                    temp[index] = word[3:len(word)-2]

            temp = [w for w in temp if w.isalpha()]
            self.trainData.append(temp)

    def _loadModel_2(self):
        if not os.path.exists('GoogleNews-vectors-negative300.bin'):
            raise ValueError("SKIP: You need to download the google news model")

        self._model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        self._model.init_sims(replace=True)

    def calculate_distance_2(self,s1,s2):
        distance = self._model.wmdistance(s1,s2)
        return distance

    def save_data(self):
        for i in range(self.n):
            self.temp[i] = np.where(self.result == i)

        f_dict = open('cache/data/labels.txt','wb')
        pickle.dump(self.temp,f_dict)
        f_dict.close()
        print('save file successfully ...')

    def load_data(self):
        f_dict = open('cache/data/labels.txt','rb')

        if len(self.temp) == 0:
            self.temp = pickle.load(f_dict)
        f_dict.close()
        print('load file successfully ...')

if __name__ == '__main__':

    if len(sys.argv) > 1:
        label_num = int(sys.argv[1])
        type_ = sys.argv[2]
        c = Classify(n=label_num,type=type_)
        c.classify()
    else:
        c = Classify()
        c.classify()












