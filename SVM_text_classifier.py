import os
import re
import string
import math
from decimal import Decimal
# use natural language toolkit
import pandas as pd
import nltk
#from nltk.stem.lancaster import LancasterStemmer
## word stemmer
#stemmer = LancasterStemmer()

from nltk.stem.isri import ISRIStemmer

stemmer=ISRIStemmer()
#import chardet
#with open('6000.csv', 'rb') as f:
#    result = chardet.detect(f.read())  # or readline if the file is large
dataset = pd.read_excel('6000.xlsx', encoding ='utf-8-sig')
dataset=dataset.dropna()
dataset=dataset.reset_index(drop=True)

x=dataset.iloc[:,0]
y=dataset.iloc[:,1]
X=x.to_dict()

X=[]
for d in range(len(x)):
    b=x[d].lower()
    X.append(b)
   

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
count_vect=CountVectorizer()
a=count_vect.fit_transform(X)
a.toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.02, random_state = 0)

count_vect=CountVectorizer()
X_train_counts=count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.toarray()


from imblearn.over_sampling import RandomOverSampler
sm=RandomOverSampler()
X_train_res, y_train_res = sm.fit_sample(X_train_tfidf, y_train)

#unique, counts = np.unique(y_train_res, return_counts=True)
#print(list(zip(unique, counts)))

from sklearn.svm import SVC

clf= SVC(kernel = 'rbf', random_state = 0)
clf.fit(X_train_res, y_train_res)
clf.score(X_train_res, y_train_res)




X_test_tfidf=count_vect.transform(X_test)

y_pred=clf.predict(X_test_tfidf)


from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
Accuracy_Score = accuracy_score(y_test, y_pred)

#from sklearn.datasets import load_svmlight_files
#X_train, y_train, X_test, y_test = load_svmlight_files(['/path-to-file/train.txt', '/path-to-file/test.txt'])