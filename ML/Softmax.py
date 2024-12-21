import numpy as np
import os
import pickle

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, recall_score, \
    precision_score
import matplotlib.pyplot as plt

data_dir = 'cifar-10-python/cifar-10-batches-py'
# 加载cifar-10
def load_cifar10_batch(file):
    with open(file, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1') # 使用pickle来反序列化数据，并将图像数据从原始格式转换为NumPy数组
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float") # 将图像从(channels, height, width)格式转换为(height, width, channels)格式
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
'''
print(X_batch)
print(Y_train) #5个数组，每个数组里10000个标签
'''

# 合并训练数据
X_train = np.concatenate(X_train)
Y_train = np.concatenate(Y_train)

# 加载测试数据
file = os.path.join(data_dir, 'test_batch')
X_test, Y_test = load_cifar10_batch(file)

# 将标签从0-9转换为1-10
Y_train = Y_train + 1
Y_test = Y_test + 1

# 划分训练集和测试集，将图像数据从三维展平成二维
X_train_flat = X_train.reshape(X_train.shape[0], -1)  # 将图像展平
X_test_flat = X_test.reshape(X_test.shape[0], -1)  # 将图像展平
'''
print(X_train_flat.shape)
(50000, 3072)
'''

# 特征缩放，使用StandardScaler对特征进行标准化处理，使得每个特征的均值为0，标准差为1
scaler = StandardScaler()
X_train_flat = scaler.fit_transform(X_train_flat)
X_test_flat = scaler.transform(X_test_flat)

# 将标签二值化
Y_train_bin = label_binarize(Y_train, classes=np.arange(1, 11))
Y_test_bin = label_binarize(Y_test, classes=np.arange(1, 11))
'''
print(Y_train_bin[0])
[0 0 0 0 0 0 1 0 0 0] #one-hot
'''

# 定义损失函数
def compute_loss(y_true, y_pred):
    return log_loss(y_true, y_pred)

# 初始化权重和偏置
weights = np.zeros((X_train_flat.shape[1], Y_train_bin.shape[1]))
bias = np.zeros(Y_train_bin.shape[1])

# 学习率
learning_rate = 0.01

# 小批量大小
batch_size = 50000

# 训练模型
num_iterations = 2000
loss_values = []

for i in range(num_iterations):
    # 随机选择一个小批量
    batch_indices = np.random.choice(X_train_flat.shape[0], batch_size, replace=False)
    X_batch = X_train_flat[batch_indices]
    Y_batch = Y_train_bin[batch_indices]

    # 预测
    linear_model = np.dot(X_batch, weights) + bias

    # 使用softmax函数将线性模型的输出转换为概率
    def softmax(z):
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # 减去最大值以提高数值稳定性
        return e_z / np.sum(e_z, axis=1, keepdims=True)

    y_predicted_proba = softmax(linear_model)

    # 计算损失
    loss = compute_loss(Y_batch, y_predicted_proba)
    loss_values.append(loss)

    # 计算梯度
    dw = np.dot(X_batch.T, (y_predicted_proba - Y_batch)) / batch_size
    db = np.sum(y_predicted_proba - Y_batch, axis=0) / batch_size

    # 更新权重和偏置
    weights -= learning_rate * dw
    bias -= learning_rate * db

    if (i + 1) % 100 == 0:
        print(f'Iteration {i + 1}, Loss: {loss:.4f}')

# 训练完成后进行预测
linear_model = np.dot(X_test_flat, weights) + bias
y_predicted_proba = softmax(linear_model)
Y_pred = np.argmax(y_predicted_proba, axis=1) # 使用np.argmax函数找到概率分布中最大值的索引，即模型预测的类别

# 评估模型
accuracy_score_value = accuracy_score(Y_test, Y_pred+1)
recall_score_value = recall_score(Y_test, Y_pred+1, average='macro')
precision_score_value = precision_score(Y_test, Y_pred+1, average='macro')
classification_report_value = classification_report(Y_test, Y_pred+1)
conf_matrix = confusion_matrix(Y_test, Y_pred+1)

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
plt.xticks(tick_marks, range(1, 11))
plt.yticks(tick_marks, range(1, 11))
thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# 可视化损失曲线
plt.plot(loss_values, label='Training Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.show()
