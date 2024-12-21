import os
import cv2
import math
import time
import numpy as np
import tqdm
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, precision_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class Classifier(object):
    def __init__(self, filePath):
        self.filePath = filePath

    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def get_data(self):
        TrainData = []
        TestData = []
        for childDir in os.listdir(self.filePath):
            if 'data_batch' in childDir:
                f = os.path.join(self.filePath, childDir)
                data = self.unpickle(f)
                train = np.reshape(data[b'data'], (10000, 3, 32 * 32))
                labels = np.reshape(data[b'labels'], (10000, 1))
                fileNames = np.reshape(data[b'filenames'], (10000, 1))
                datalebels = zip(train, labels, fileNames)
                TrainData.extend(datalebels)
            if childDir == "test_batch":
                f = os.path.join(self.filePath, childDir)
                data = self.unpickle(f)
                test = np.reshape(data[b'data'], (10000, 3, 32 * 32))
                labels = np.reshape(data[b'labels'], (10000, 1))
                fileNames = np.reshape(data[b'filenames'], (10000, 1))
                TestData.extend(zip(test, labels, fileNames))
        print("data read finished!")
        return TrainData, TestData

    def get_hog_feat(self, image, stride=8, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        cx, cy = pixels_per_cell
        bx, by = cells_per_block
        sx, sy = image.shape
        n_cellsx = int(np.floor(sx // cx))  # number of cells in x
        n_cellsy = int(np.floor(sy // cy))  # number of cells in y
        n_blocksx = (n_cellsx - bx) + 1
        n_blocksy = (n_cellsy - by) + 1
        gx = np.zeros((sx, sy), dtype=np.float32)
        gy = np.zeros((sx, sy), dtype=np.float32)
        eps = 1e-5
        grad = np.zeros((sx, sy, 2), dtype=np.float32)
        for i in range(1, sx - 1):
            for j in range(1, sy - 1):
                gx[i, j] = image[i, j - 1] - image[i, j + 1]
                gy[i, j] = image[i + 1, j] - image[i - 1, j]
                grad[i, j, 0] = np.arctan(gy[i, j] / (gx[i, j] + eps)) * 180 / math.pi
                if gx[i, j] < 0:
                    grad[i, j, 0] += 180
                grad[i, j, 0] = (grad[i, j, 0] + 360) % 360
                grad[i, j, 1] = np.sqrt(gy[i, j] ** 2 + gx[i, j] ** 2)
        normalised_blocks = np.zeros((n_blocksy, n_blocksx, by * bx * orientations))
        for y in range(n_blocksy):
            for x in range(n_blocksx):
                block = grad[y * stride:y * stride + 16, x * stride:x * stride + 16]
                hist_block = np.zeros(32, dtype=np.float32)
                eps = 1e-5
                for k in range(by):
                    for m in range(bx):
                        cell = block[k * 8:(k + 1) * 8, m * 8:(m + 1) * 8]
                        hist_cell = np.zeros(8, dtype=np.float32)
                        for i in range(cy):
                            for j in range(cx):
                                n = int(cell[i, j, 0] / 45)
                                hist_cell[n] += cell[i, j, 1]
                        hist_block[(k * bx + m) * orientations:(k * bx + m + 1) * orientations] = hist_cell[:]
                normalised_blocks[y, x, :] = hist_block / np.sqrt(hist_block.sum() ** 2 + eps)
        return normalised_blocks.ravel()

    def get_feat(self, TrainData, TestData):
        train_feat = []
        test_feat = []
        for data in tqdm.tqdm(TestData):
            image = np.reshape(data[0].T, (32, 32, 3))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.
            fd = self.get_hog_feat(gray)
            fd = np.concatenate((fd, data[1]))
            test_feat.append(fd)
        test_feat = np.array(test_feat)
        np.save("test_feat.npy", test_feat)
        print("Test features are extracted and saved.")
        for data in tqdm.tqdm(TrainData):
            image = np.reshape(data[0].T, (32, 32, 3))
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.
            fd = self.get_hog_feat(gray)
            fd = np.concatenate((fd, data[1]))
            train_feat.append(fd)
        train_feat = np.array(train_feat)
        np.save("train_feat.npy", train_feat)
        print("Train features are extracted and saved.")
        return train_feat, test_feat

    def classification(self, train_feat, test_feat):
        t0 = time.time()
        clf = LinearSVC()
        print("Training a Linear SVM Classifier.")
        clf.fit(train_feat[:, :-1], train_feat[:, -1])
        predict_result = clf.predict(test_feat[:, :-1])
        accuracy = accuracy_score(test_feat[:, -1], predict_result)
        recall = recall_score(test_feat[:, -1], predict_result, average='macro')
        precision = precision_score(test_feat[:, -1], predict_result, average='macro')
        report = classification_report(test_feat[:, -1], predict_result)
        conf_mat = confusion_matrix(test_feat[:, -1], predict_result)

        print('The classification accuracy is %f' % accuracy)
        print('The recall score is %f' % recall)
        print('The precision score is %f' % precision)
        print('Classification report:\n%s' % report)

        # Visualize confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(10)
        plt.xticks(tick_marks, range(10))
        plt.yticks(tick_marks, range(10))
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        for i in range(conf_mat.shape[0]):
            for j in range(conf_mat.shape[1]):
                plt.text(j, i, format(conf_mat[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if conf_mat[i, j] > conf_mat.max() / 2. else "black")
        plt.show()

        t1 = time.time()
        print('The cast of time is :%f' % (t1 - t0))

    def run(self):
        if os.path.exists("train_feat.npy") and os.path.exists("test_feat.npy"):
            train_feat = np.load("train_feat.npy")
            test_feat = np.load("test_feat.npy")
        else:
            TrainData, TestData = self.get_data()
            train_feat, test_feat = self.get_feat(TrainData, TestData)
        self.classification(train_feat, test_feat)


if __name__ == '__main__':
    filePath = r'cifar-10-python/cifar-10-batches-py'
    cf = Classifier(filePath)
    cf.run()