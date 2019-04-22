from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import utils

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

    clf = KMeans(n_clusters = k,init='k-means++',max_iter=300,)
    s = clf.fit(weight)
    order_centroids = clf.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    print("第" + str(k) + "次聚类\n")
    for ss in range(k):
            print("\nCluster %d:" % ss, end='')
            for ind in order_centroids[ss, :10]:
                print(' %s' % terms[ind], end='')


    # train data classification
    # category=clf.predict(train_data)
    # print('classification results:',category)
#
# def predict(test_data):
#     pred=clf.predict(test_data)
#     print('prediction results：',pred)
#     print('similar elments:',train_data[category==pred])

# train data labels
    label=[]
    for i in range(1,len(clf.labels_)):
        label.append(clf.labels_[i-1])

    print(clf.inertia_)
    y_pred=clf.labels_


from sklearn.decomposition import  PCA
pca=PCA(n_components=2)
newData=pca.fit_transform(weight)


def generate_coor(newData,y_pred,clusters):
    x = [n[0] for n in newData]
    y = [n[1] for n in newData]

    for i in range(clusters):
        tmpx,tmpy=[],[]







#
