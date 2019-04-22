from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import utils
train_data = []
tfidfdict = {}


train_data,test_data=utils.get_corpus()

vectorizer=CountVectorizer()
transformer=TfidfTransformer()
tfidf=transformer.fit_transform(vectorizer.fit_transform(train_data))
word=vectorizer.get_feature_names()
weight=tfidf.toarray()

for i in range(len(weight)):
    for j in range(len(word)):
        getword = word[j]
        getvalue = weight[i][j]
K = range(5,20)
for k in K:

    clf = KMeans(n_clusters = k)
    s = clf.fit(weight)
    order_centroids = clf.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    print('\n'+str(k)+'类:\n')
    for ss in range(k):
            print("\nCluster %d:" % ss, end='')
            for ind in order_centroids[ss, :10]:
                print(' %s' % terms[ind], end='')

