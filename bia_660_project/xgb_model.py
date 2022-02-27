'''
Created by Vivek Gupta on 01-12-2020
'''
import pickle
import re
from string import punctuation

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import MaxAbsScaler


def load_data(df):
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


# Load data
df = pd.read_csv("data/data.csv")
X, y = load_data(df)

# Stratified Sampling
train_X, test_X, train_y, test_y = None, None, None, None
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
sss.get_n_splits(X, y)
for train_index, test_index in sss.split(X, y):
    train_X, test_X = [X[index] for index in train_index], [X[index] for index in test_index]
    train_y, test_y = [y[index] for index in train_index], [y[index] for index in test_index]

counter = CountVectorizer(stop_words=stopwords.words('english'))
counter.fit(train_X)

# Save count vectorizer
with open('save/counter.pkl', 'wb') as file:
    pickle.dump(counter, file)

# count the number of times each term appears in a document and transform each doc into a count vector
counts_train = counter.transform(train_X)  # transform the training data
counts_test = counter.transform(test_X)  # transform the test data

scaler = MaxAbsScaler()
scaler.fit(counts_train)
counts_train = scaler.transform(counts_train)
counts_test = scaler.transform(counts_test)

# Save scaler
with open('save/scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

clf = XGBClassifier(num_class=3, learning_rate=0.01, n_estimators=1000, max_depth=5,
                    min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, objective='multi:softmax', seed=27)

clf.fit(counts_train, train_y)

# Train
pred = clf.predict(counts_train)
print(accuracy_score(pred, train_y))

# Test
pred = clf.predict(counts_test)
print(accuracy_score(pred, test_y))

# Save the model
with open('save/final_model.pkl', 'wb') as file:
    pickle.dump(clf, file)
