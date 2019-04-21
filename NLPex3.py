#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:48:43 2019

@author: wangkewei
"""

__authors__ = ['Jingyi LI','Kewei WANG', 'Zhipeng CHEN', 'Wenhan ZHAO']
__emails__  = ['jingyi.li@essec.edu', 'kewei.wang@essec.edu', 'zhipeng.chen@essec.edu', 'wenhan.zhao@essec.edu']

import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np
from sklearn.preprocessing import normalize
import pandas as pd
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#This def is to load the sentences in the form of lower letter
def loadsentence(path):
    sentence = []
    with open(path) as f:
        for l in f:
            sentence.append( l.lower() )
    return sentence

#This def is to divide every chi-chat dialogues
def dialogues(train):
    seppoint = []
    for i in range(len(train)) :
        if (train[i][0:14] == "1 your persona"):
            seppoint.append(i)
    seppoint.append(len(train))
    dialogues = []
    for i in range(len(seppoint)-1):
        dialogues.append(train[seppoint[i]:seppoint[i+1]])
    return dialogues

#This def is to find the context among the dialogues
def context(dia):
    seppoint = []
    for i in range(len(dia)):
        if (dia[i][2:19] == "partner's persona" or dia[i][3:20] == "partner's persona" ):
            seppoint.append(i)
    context = dia[0:(seppoint[-1]+1)]
    return context

#This def is to remove the '\n' at the end of each sentence and 
#extract the dialogues of each person without the the illustration of character
def dialogues_extract(context):
    context_new = []
    for l in context:
        l = l.strip('\n')
        l = ' '.join([w for w in l.split()[3:]])
        context_new.append(l)
        
    context_new = ' '.join(context_new)    
    return context_new

#This def is to remove the digit in the data
def remove_digit(sentence):
    no_digital = [w for w in sentence if w.isdigit()==False]
    no_digital =''.join(no_digital)
    return no_digital

#This def is to combine the dataset from Context and Utterance
def combset(context,sentence):
    context_new = []
    context_new.append(context)
    context_new.append(sentence)
    context_new = ' '.join(context_new) 
    return context_new

#This step is to do pre-processing work
def dataprocessing(data):
    # Remove puntuation
    table = str.maketrans('', '', string.punctuation)
    new_train = [w.translate(table) for w in data] 
    # Split words
    new_train_bis=[]
    for i in range(len(data)):
        new_train_bis.append(word_tokenize(new_train[i])) 
    #Slove stop Words
    stop_words = stopwords.words('english')
    train_=[]
    for i in range(len(new_train_bis)):
        train_.append(' '.join([w for w in new_train_bis[i] if not w in stop_words]))       
    #Stem
    porter = PorterStemmer()
    stemmed=[]
    for i in range(len(train_)):
        stemmed.append(','.join([porter.stem(word) for word in word_tokenize(train_[i])]))        
    return stemmed

#This step is to build the trainset
def trainset(text):
    train = loadsentence(text)#text must be the path of text
    train = dialogues(train)
    trainset=[]
    n = 200
    for i in range(n):
        dial = train[i]
        n1 = len(dial)
        
        cont = context(dial)
        n2 = len(cont)
        
        cont = dialogues_extract(cont)
        
        for j in range(n2,n1):
            distr = dial[j].split("\t")
            cor = distr[1]
            utt = distr[0]         
            cont = combset(cont,remove_digit(utt))
            trainset.append({'Context': cont, 'Utterance': cor, 'Label': 1})
            answers = distr[3].split("|") 
        
            for i in range(len(answers)-1):
                d = {'Context': cont, 'Utterance': answers[i], 'Label': 0}
                trainset.append(d)  
                
            cont = combset(cont,cor) 
    
    trainset_raw = pd.DataFrame(data = trainset)
    df_train = pd.DataFrame()
    df_train['Context']= dataprocessing(trainset_raw['Context'])
    df_train['Utterance']= dataprocessing(trainset_raw['Utterance'])
    df_train['Label']=trainset_raw['Label']
    
    return df_train
    

#This step is to build the testnset   
def testset2(text):    
    test = loadsentence(text)#text must be the path of text
    test = dialogues(test)
    testset=[]
    n=200
    for i in range(n):
        dial = test[i]
        n_d = len(dial)
        
        cont = context(dial)
        n_s = len(cont)
        
        cont = dialogues_extract(cont)
       
        for j in range(n_s,n_d):
            distr = dial[j].split("\t")
            utt = distr[0]
            answers = distr[3].split("|")    
            n_a = len(answers)
            cont = combset(cont,remove_digit(utt))
            
            #create dictionary for row2
            list_of_key = [x for x in range(1,n_a+1)]
            list_of_key.append('context')
            
            list_of_values = [answers[x] for x in range(n_a)]
            list_of_values.append(cont)
            
            dic = dict( zip(list_of_key,list_of_values ))
            testset.append(dic)
            
    df_test = pd.DataFrame(data = testset)
    for i in range(df_test.shape[1]):
        df_test.iloc[:,i]= dataprocessing(df_test.iloc[:,i])
    
    return testset, df_test

#This def is used to evaluate the accuracy
def evaluate_recall(y, y_test, k=1):
    num_examples = float(len(y))
    num_correct = 0
    for predictions, label in zip(y, y_test):
        if label in predictions[:k]:
            num_correct += 1
    return num_correct/num_examples

#This def is to return the result of raw data
def retrieve_sentence(y_pred,df_test): 
    l = []
    for i in range(len(y_pred)) :
        l.append([y_pred[i][0]+1,df_test.iloc[i,:df_test.shape[1]-1][y_pred[i][0]+1]])
    return l

#Set the TFIDF model part including the save model and load model
class TFIDFPredictor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def train(self, data):
        self.vectorizer.fit(np.append(data.Context.values,data.Utterance.values))

    def predict(self, context, utterances):
        # Convert context and utterances into tfidf vector
        vector_context = self.vectorizer.transform([context])
        vector_doc = self.vectorizer.transform(utterances)
        # The dot product measures the similarity of the resulting vectors
        result = np.dot(vector_doc, vector_context.T).todense()
        result = np.asarray(result).flatten()
        # Sort by top results and return the indices in descending order
        return np.argsort(result, axis=0)[::-1]
    
    def save(self,path):
        with open(path,'wb') as fout:
            pickle.dump(self.vectorizer,fout)
    
    def load(self,path):
        with open(path,'rb') as f:
            self.vectorizer = pickle.load(f)
            
            
            
            
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true')
    group.add_argument('--test', action='store_true')
    
    parser.add_argument('--model', help='path to model file (for saving/loading)', required=True)
    parser.add_argument('--text', help='path to text file (for training/testing)', required=True)
    opts = parser.parse_args()
    
    pred = TFIDFPredictor()
    
    if opts.train:
        df_train = trainset(opts.text)
        pred.train(df_train)
        pred.save(opts.model)
    else:
        assert opts.test,opts.test
        testset, df_test = testset2(opts.text)
        df_test_raw = pd.DataFrame(data = testset)
        
        pred.load(opts.model)
        y_test = np.zeros(len(df_test))
        y = [pred.predict(df_test.context[x], df_test.iloc[x,:df_test.shape[1]-1].values) for x in range(len(df_test))]
        l = retrieve_sentence(y,df_test_raw)
        for i in range(len(l)):
            print(l[i])        
        
        
        

    
    


    
    










