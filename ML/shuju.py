import numpy as np
import cv2

# 打开cifar-10数据集文件目录
def unpickle(file):
    import pickle
    with open("cifar-10-python/cifar-10-batches-py/"+file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#打开cifar-10文件的data_batch_1
data_batch=unpickle("data_batch_1")
# data_batch为字典，包含四个字典键：
# b'batch_label'
# b'labels' 标签
# b'data'  图片像素值
# b'filenames'
print(data_batch)
print(len(b'labels'))

cifar_label=data_batch[b'labels']
cifar_data=data_batch[b'data']

#把字典的值转成array格式，方便操作
cifar_label=np.array(cifar_label)
print(cifar_label.shape)
cifar_data=np.array(cifar_data)
print(cifar_data.shape)
