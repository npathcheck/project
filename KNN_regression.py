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
        print(len(self.test_datas))

    def write(self, file_path):
        IO_regression.write(file_path, self.predict_labels)

    def calculate_euclidean(self, data1, data2, tag1, tag2):
        return 1 / ((np.linalg.norm((data1-data2))**2 + len(tag1^tag2))**0.5 + 1e-6)

    def calculate_cos(self, data1, data2, tag1, tag2):
        tags = list(tag1 | tag2)
        vec1 = np.zeros(len(tags))
        vec2 = np.zeros(len(tags))
        for i in range(len(tags)):
            if tags[i] in tag1:
                vec1[i] = 1
            if tags[i] in tag2:
                vec2[i] = 1
        data1 = np.c_[data1, vec1]
        data2 = np.c_[data2, vec2]
        cos = np.dot(data1, data2) / (np.linalg.norm(data1) * np.linalg.norm(data2))
        return 1 / -math.log((cos+1)/2)

    def k_nearest_neighbors(self, k):
        self.predict_labels = []
        for i in range(len(self.test_datas)):
            print("step: " + str(i))
            ndistance = []
            for j in range(len(self.train_datas)):
                distance = self.calculate_euclidean(self.test_datas[i],self.train_datas[j],self.test_tags[i],self.train_tags[j])
                ndistance.append([distance, self.train_labels[j]])
            knearest = np.array([[distance[0], distance[1]*distance[0]] for distance in heapq.nsmallest(k, ndistance)])
            self.predict_labels.append(np.sum(knearest[:,1])/np.sum(knearest[:,0]))



if __name__ == '__main__':
    knn = KNN()
    knn.read("data/回归")
    knn.k_nearest_neighbors(250)
