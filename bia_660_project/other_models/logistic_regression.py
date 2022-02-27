'''
Created by Vivek Gupta on 12-12-2020
'''

import re
from string import punctuation

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit


def loadData(df):
    desc = []
    label = []
    for i in df.values:
        text = i[0]
        text = text.lower()
        text = text.strip()
        # Removing numbers, *** and www links from the data
        text = re.sub('[0-9]+\S+|\s\d+\s|\w+[0-9]+|\w+[\*]+.*|\s[\*]+\s|www\.[^\s]+', '', text)
        # Removing punctuation
        for p in punctuation:
            text = text.replace(p, ' ')

        # Removing labels from the description
        if i[1] == "Data Scientist":
            text = re.sub('data sci[a-z]+', ' ', text, re.I)
        elif i[1] == "Data Engineer":
            text = re.sub('data eng[a-z]+', ' ', text, re.I)
        else:
            text = re.sub('software eng[a-z]+', ' ', text, re.I)

        my_tokens = text.split()
        porter = PorterStemmer()
        stemmed_tokens = []
        for token in my_tokens:
            if len(token) > 2:
                stemmed_tokens.append(porter.stem(token))

        text = " ".join(stemmed_tokens)
        desc.append(text)
        label.append(i[1])
    return desc, label


df = pd.read_csv("../data/data.csv")
X, y = loadData(df)

# train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.7, random_state=42)

train_X, test_X, train_y, test_y = None, None, None, None
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
sss.get_n_splits(X, y)
for train_index, test_index in sss.split(X, y):
    train_X, test_X = [X[index] for index in train_index], [X[index] for index in test_index]
    train_y, test_y = [y[index] for index in train_index], [y[index] for index in test_index]

counter = CountVectorizer(stop_words=stopwords.words('english'))
counter.fit(train_X)

# count the number of times each term appears in a document and transform each doc into a count vector
counts_train = counter.transform(train_X)  # transform the training data
counts_test = counter.transform(test_X)  # transform the test data

LREG_classifier = LogisticRegression()


# # build the parameter grid
LREG_grid = [{'C': [0.5, 1, 1.5, 2], 'penalty': ['l2'], 'max_iter': [100, 150, 200]}]

# build a grid search to find the best parameters
gridsearchLREG = GridSearchCV(LREG_classifier, LREG_grid, cv=3, n_jobs=-1)

# run the grid search
gridsearchLREG.fit(counts_train, train_y)

# use the VT classifier to predict
predicted = gridsearchLREG.predict(counts_test)

# print the accuracy
print(accuracy_score(predicted, test_y))
