import numpy as np
import os
import pickle

from sklearn import naive_bayes
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score
import matplotlib.pyplot as plt

# 假设数据集文件路径
data_dir = 'cifar-10-python/cifar-10-batches-py'
# 加载cifar-10
def load_cifar10_batch(file):
    with open(file, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1') # 使用pickle来反序列化数据
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float") # 转换图像格式
        Y = np.array(Y)
        return X, Y

# 加载所有数据中的5个训练批次，并将它们合并成一个完整的训练集
X_train = []
Y_train = []
for i in range(1, 6):
    file = os.path.join(data_dir, f'data_batch_{i}')
    X_batch, Y_batch = load_cifar10_batch(file)
    X_train.append(X_batch)
    Y_train.append(Y_batch)

# 合并训练数据
X_train = np.concatenate(X_train)
Y_train = np.concatenate(Y_train)

# 加载测试数据
file = os.path.join(data_dir, 'test_batch')
X_test, Y_test = load_cifar10_batch(file)

# 划分训练集和测试集，将图像数据从三维展平成二维
X_train_flat = X_train.reshape(X_train.shape[0], -1)  # 将图像展平
X_test_flat = X_test.reshape(X_test.shape[0], -1)  # 将图像展平

# 特征缩放，使用StandardScaler对特征进行标准化处理
scaler = StandardScaler()
X_train_flat = scaler.fit_transform(X_train_flat)
X_test_flat = scaler.transform(X_test_flat)

# 使用PCA进行特征降维
pca = PCA(n_components=0.75)  # 保留95%的方差
X_train_pca = pca.fit_transform(X_train_flat)
X_test_pca = pca.transform(X_test_flat)

# 高斯朴素贝叶斯假设特征之间相互独立，并且每个特征都服从高斯分布
gbn = naive_bayes.GaussianNB()
gbn.fit(X_train_pca, Y_train)

# 训练完成后进行预测
Y_pred = gbn.predict(X_test_pca)

# 评估模型
accuracy_score_value = accuracy_score(Y_test, Y_pred)
recall_score_value = recall_score(Y_test, Y_pred, average='macro')
precision_score_value = precision_score(Y_test, Y_pred, average='macro')
classification_report_value = classification_report(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)

# 打印结果
print("准确率：", accuracy_score_value)
print("召回率：", recall_score_value)
print("精确率：", precision_score_value)
print("分类报告：\n", classification_report_value)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()

# 在每个方格内显示数字
tick_marks = np.arange(10)
plt.xticks(tick_marks, range(10))
plt.yticks(tick_marks, range(10))
thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()