'''
导入必要的库：csv 用于读取 CSV 文件，tensorflow 用于构建和训练模型，
numpy 用于数值计算，pandas 用于数据处理，matplotlib.pyplot 用于绘图，
mean_absolute_error 和 mean_squared_error 用于计算评估指标。
'''
import csv
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 读取数据,使用 pandas 读取 CSV 文件 bbc-text.csv，显示前几行数据，并打印数据集中所有唯一的类别。
'''
输出结果：
0                tech
1            business
2               sport
3               sport
4       entertainment
            ...      
2220         business
2221         politics
2222    entertainment
2223         politics
2224            sport
Name: category, Length: 2225, dtype: object
'''
data = pd.read_csv("bbc-text.csv")
data.head()
np.unique(data.category)

# 设置模型参数
vocab_size = 1000  # 词汇表大小
embedding_dim = 16  # 嵌入维度
max_length = 120  # 最大序列长度
trunc_type = 'post'  # 截断和填充类型
padding_type = 'post'
oov_tok = ""  # 未出现在词汇表中的单词的标记
training_portion = 0.8  # 训练数据占总数据的比例

# 初始化句子和标签列表
sentences = []
labels = []
# 定义一个停用词列表，这些词将从文本中移除
stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]

# 手动处理文本数据
'''
打开 CSV 文件，逐行读取数据。跳过标题行，将每一行的第一个字段作为标签，
第二个字段作为句子。移除句子中的停用词，并将句子拆分为单词列表。
'''
with open("bbc-text.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)  # 跳过标题行 (category,text)
    for row in reader:
        labels.append(row[0])
        sentence = row[1]
        for word in stopwords:
            token = " " + word + " "
            sentence = sentence.replace(token, " ")
        sentences.append(sentence.split())  # 将句子拆分为单词

# 创建词汇表并映射单词到索引
word_index = {}
index = 1
for sentence in sentences:
    for word in sentence:
        if word not in word_index and index < vocab_size:
            word_index[word] = index
            index += 1

# 将句子转换为索引序列
def sentence_to_indices(sentence):
    return [word_index.get(word, 0) for word in sentence if word in word_index]

# 计算训练集的大小=2225*0.8,4/5为训练集，1/5为测试集
train_size = int(len(sentences) * training_portion)

# 将前 train_size 个句子和标签作为训练数据
train_sentences = sentences[:train_size]
train_labels = labels[:train_size]

# 将剩余的句子和标签作为验证数据
validation_sentences = sentences[train_size:]
validation_labels = labels[train_size:]

# 将训练集和验证集的句子转换为整数索引序列
train_sequences = [sentence_to_indices(sentence) for sentence in train_sentences]
validation_sequences = [sentence_to_indices(sentence) for sentence in validation_sentences]

# 定义一个函数用于填充或截断序列，以确保所有序列具有相同的长度。
def pad_sequences(sequences, maxlen):
    padded_sequences = []
    for sequence in sequences:
        if len(sequence) > maxlen:
            padded_sequences.append(sequence[:maxlen])
        else:
            padded_sequences.append(sequence + [0] * (maxlen - len(sequence)))
    return padded_sequences

# 对训练集和验证集的序列进行填充或截断
train_padded = pad_sequences(train_sequences, max_length)
validation_padded = pad_sequences(validation_sequences, max_length)

# 将标签转换为数值索引
# print(label_index)后的输出为{'business': 0, 'entertainment': 1, 'sport': 2, 'tech': 3, 'politics': 4}
label_index = {}
index = 0
for label in set(labels):
    label_index[label] = index
    index += 1

# 将训练集和验证集的标签转换为整数索引数组
training_label_seq = np.array([label_index[label] for label in train_labels])
validation_label_seq = np.array([label_index[label] for label in validation_labels])

# 构建模型，包括嵌入层、全局平均池化层和两个全连接层，特征提取和分类
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length), # 嵌入层：用于将正整数（通常是单词索引）转换为密集向量
    tf.keras.layers.GlobalAveragePooling1D(), # 全局平均池化层：用于对嵌入层的输出进行全局平均池化，进行特征提取
    tf.keras.layers.Dense(42, activation='relu'), # 第一个全连接层：为模型引入非线性，帮助学习复杂模式
    tf.keras.layers.Dense(len(label_index), activation='softmax') # 第二个全连接层：使用Softmax激活函数，将输出转换为概率分布，每个类别一个概率值
])
# 编译模型，指定损失函数、优化器和评估指标，使用稀疏分类交叉熵作为损失函数，Adam优化器
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 打印模型的摘要信息
model.summary()

# 训练模型，并记录历史
num_epochs = 30
history = model.fit(np.array(train_padded), training_label_seq, epochs=num_epochs,
                    validation_data=(np.array(validation_padded), validation_label_seq), verbose=2)

# 绘制训练和验证准确率和损失曲线
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

# 解码句子，创建一个从整数索引到单词的反向映射
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# 定义一个函数来解码整数索引序列回文本
def decode_sentence(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# 获取嵌入层的权重
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)  # 预期输出：(1000, 16)

# 使用模型进行预测,五类每类输出一个值
validation_predictions = model.predict(np.array(validation_padded))

# 将预测结果转换为类别标签
predicted_labels = np.argmax(validation_predictions, axis=1)
print(predicted_labels)

# 计算评估指标
mae = mean_absolute_error(validation_label_seq, predicted_labels)
mse = mean_squared_error(validation_label_seq, predicted_labels)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
