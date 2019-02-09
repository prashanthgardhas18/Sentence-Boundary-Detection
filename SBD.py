#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 22:34:28 2019

@author: prashanthkumargardhas
"""
import pandas as pd
import numpy as np
import nltk
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
nltk.download('stopwords')
from nltk.corpus import stopwords

#This function extracts the features from the train and test datasets
def get_feature_vectors(file_name):
        with open(file_name) as data_file:
            lines = data_file.readlines()
        data = pd.DataFrame([], columns = ['Left_Word', 'Right_Word', 'L_Length < 3', 'L_Is_Cap', 'R_Is_Cap', 'L_StopWord' , 'R_Stopword', 'R_Length < 3', 'Target_Label'])
    
        for i in range (len(lines)):
            l = lines[i]
            
            if l.find('.') >= 0:
                words = lines[i].split()
                
                if words[1] == '.' or words[2] == 'TOK':
                    continue
                if len(words[1])>0:
                    left = words[1][:len(words[1])-1]
       
                target = words[2]
                
                if i == len(lines)-1:     #for the Last line, right side of period is nothing, so assigning Empty space
                    right = " "
                else:
                    nextline = lines[i+1].split()
                    right = nextline[1]
                     
                if len(left) < 3:
                    llen = 1
                else:
                    llen = 0
                    
                if left[0].isupper():
                    L_upper = 1
                else:
                    L_upper = 0
                    
                if right[0].isupper():
                    R_upper = 1
                else:
                    R_upper = 0
                
                # 3 Additional Features 
                # 1) Check Left word is stopword
                # 2) Check Right word is stopword
                # 3) Right word is less than 3
                
                if left in stopwords.words('english'):
                    lstopword = 1
                else:
                    lstopword = 0
                
                if right in stopwords.words('english'):
                    rstopword = 1
                else:
                    rstopword = 0
                    
                if len(right) < 3:
                    Rlen = 1
                else:
                    Rlen = 0 
                
                
                #print (words, nextline, left, target, right)
                
                ser = pd.Series([left, right, llen, L_upper, R_upper, lstopword, rstopword, Rlen,  target], index = ['Left_Word', 'Right_Word', 'L_Length < 3', 'L_Is_Cap', 'R_Is_Cap', 'L_StopWord' , 'R_Stopword', 'R_Length < 3', 'Target_Label'])
                data = data.append(ser, ignore_index=True)
        return data
    
train_data = get_feature_vectors(sys.argv[1])   #passing SBD.train datafile
test_data = get_feature_vectors(sys.argv[2])    #passing SBD.test datafile

#Train data        
train_data_features = train_data.values[:,2:7]
train_data_target_label = train_data.values[:,8]

#Test data
test_data_features = test_data.values[:,2:7]
test_data_target_label = test_data.values[:,8]

#print(test_data_features)
#Decision Tree Classification model on traindata  

Entropy_Classifier = DecisionTreeClassifier(criterion = "entropy").fit(train_data_features, train_data_target_label)
    
test_prediction = Entropy_Classifier.predict(test_data_features)
#print(test_prediction)

#Output file : SBD.test.out which includes first two coloumns from SBD.test file 
#and along with Label EOS or NEOS predicted by the System
SBD_test_output1 = test_data.values[:,0:2]
SBD_test_output = np.column_stack((SBD_test_output1,test_prediction))
pd.DataFrame(SBD_test_output).to_csv('/Users/prashanthkumargardhas/Spyder/SBD.test.out')

print("Accuracy is ", accuracy_score(test_data_target_label,test_prediction)*100)