import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score, accuracy_score, precision_score, recall_score, \
    classification_report, confusion_matrix
from mpl_toolkits.mplot3d import Axes3D

# 假设数据集文件路径
data_dir = 'C:/Users/18522/ML/cifar-10-python/cifar-10-batches-py'

# 加载 CIFAR-10 批次数据
def load_cifar10_batch(file):
    with open(file, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        x = datadict['data']
        y = datadict['labels']
        x = x.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        y = np.array(y)
        return x, y

# 加载所有数据中的5个训练批次，并将它们合并成一个完整的训练集
x_train = []
y_train = []
for i in range(1, 6):
    file = os.path.join(data_dir, f'data_batch_{i}')
    x_batch, y_batch = load_cifar10_batch(file)
    x_train.append(x_batch)
    y_train.append(y_batch)

# 合并训练数据
x_train = np.concatenate(x_train)
y_train = np.concatenate(y_train)

x = x_train.reshape(x_train.shape[0], -1)
y = y_train.reshape(-1)

# 加载测试数据
file = os.path.join(data_dir, 'test_batch')
x_test, y_test = load_cifar10_batch(file)

# reshape the data
x_train = x_train.reshape(x_train.shape[0], 32*32*3)
x_test = x_test.reshape(x_test.shape[0], 32*32*3)

# normalize the data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30)
x_train  = np.vstack((x_train, x_test))
X_tsne = tsne.fit_transform(x_train)


# create a KMeans object
kmeans = KMeans(n_clusters=10, random_state=0)

# fit the KMeans on the t-SNE transformed data
kmeans.fit(X_tsne)

# predict the cluster labels
y_predX_tsne = kmeans.predict(X_tsne)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_predX_tsne)
plt.xlabel('t-SNE component 1')
plt.ylabel('t-SNE component 2')
plt.title('CIFAR-10 dataset in the t-SNE 2D space with K-means clusters')
plt.show()

ss_tsne_kmeans = silhouette_score(X_tsne, y_predX_tsne)
db_tsne_kmeans = davies_bouldin_score(X_tsne, y_predX_tsne)
print(ss_tsne_kmeans)
print(db_tsne_kmeans)
