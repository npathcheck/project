import IO_regression
import numpy as np
import heapq
import math

class KNN():

    train_datas = []
    train_tags = []
    train_labels = []
    test_datas = []
    test_tags = []
    predict_labels = []

    def read(self, file_path):
        self.train_datas, self.train_tags, self.train_labels = IO_regression.read(file_path, "train")
        self.test_datas, self.test_tags = IO_regression.read(file_path, "test")
        self.zero_one_standard(self.train_datas)
        self.zero_one_standard(self.test_datas)

    def write(self, file_path):
        IO_regression.write(file_path, self.predict_labels)

    def zero_one_standard(self, data):
        for i in range(6):
            data[:,i] = (data[:,i] - np.min(data[:,i])) / (np.max(data[:,i] - np.min(data[:,i]))) * 10

    def z_score_standard(self, data):
        for i in range(6):
            data[:,i] = (data[:,i] - np.mean(data[:,i])) / np.std(data[:,i]) * 10

    def calculate_euclidean(self, data1, data2, tag1, tag2):
        return (np.linalg.norm((data1-data2))**2 + len(tag1^tag2))**0.5

    def calculate_cos(self, data1, data2, tag1, tag2):
        cos = (np.dot(data1, data2)+len(tag1&tag2)) / ((np.linalg.norm(data1)+len(tag1)) * (np.linalg.norm(data2)+len(tag1)))
        return -math.log(cos)

    def k_nearest_neighbors(self, k):
        self.predict_labels = []
        for i in range(len(self.test_datas)):
            print("step: " + str(i))
            ndistance = []
            for j in range(len(self.train_datas)):
                distance = self.calculate_cos(self.test_datas[i],self.train_datas[j],self.test_tags[i],self.train_tags[j])
                ndistance.append([distance, self.train_labels[j]])
            knearest = np.array([distance for distance in heapq.nsmallest(k, ndistance)])
            kweight = 1 / knearest[:,0]
            self.predict_labels.append(np.sum(np.dot(knearest[:,1], kweight))/np.sum(kweight))


if __name__ == '__main__':
    knn = KNN()
    knn.read("data/回归")
    knn.k_nearest_neighbors(100)
    knn.write("data/回归/16337250_3.txt")