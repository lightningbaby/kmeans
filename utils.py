from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import pandas as pd

def get_corpus():
    file_path='data/2015.csv'
    data=pd.read_csv(file_path)
    # doc=data['Document Title']

    corpus=data['Document Title'].map(str)+data['Authors'].map(str)+data['Author Keywords'].map(str)
    # print(train_data)
    train_data,test_data=train_test_split(corpus,test_size=0.2)
    return train_data,test_data

# get_corpus()