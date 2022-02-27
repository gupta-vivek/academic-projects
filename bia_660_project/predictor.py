'''
Created by Vivek Gupta
'''

import re
from string import punctuation
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MaxAbsScaler
from xgboost.sklearn import XGBClassifier
import pickle
import csv
import sys


def load_data(df):
    desc = []
    for i in df.values:
        text = i[0]
        text = text.lower()
        text = text.strip()
        # Removing numbers, *** and www links from the data
        text = re.sub('[0-9]+\S+|\s\d+\s|\w+[0-9]+|\w+[\*]+.*|\s[\*]+\s|www\.[^\s]+', '', text)

        # Removing punctuation
        for p in punctuation:
            text = text.replace(p, ' ')

        my_tokens = text.split()
        porter = PorterStemmer()
        stemmed_tokens = []
        for token in my_tokens:
            if len(token) > 2:
                stemmed_tokens.append(porter.stem(token))

        text = " ".join(stemmed_tokens)
        desc.append(text)

    return desc


# Load model
with open('save/final_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load count vectorizer
with open('save/counter.pkl', 'rb') as file:
    counter = pickle.load(file)

# Load scaler
with open('save/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Read and transform data
df = pd.read_csv(sys.argv[1])
X = load_data(df)
counts_test = counter.transform(X)
counts_test = scaler.transform(counts_test)

# Predict
pred = model.predict(counts_test)

# Save the predictions
fp = open('predictions.csv', 'w', newline='')
csv_writer = csv.writer(fp)
for p in pred:
    csv_writer.writerow([p.strip()])
fp.close()
