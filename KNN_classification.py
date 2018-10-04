import collections
import numpy as np
import heapq
import io_data

class KNN():

    train_datas = []
    train_labels = []
    test_datas = []
    k_train_datas = []
    k_train_labels = []
    k_validation_datas = []
    k_validation_labels = []

    # 读取训练集
    def read_train(self, train_data_path, train_label_path):
        self.train_datas, self.train_labels = io_data.read_train(train_data_path, train_label_path)
        for i in range(len(self.train_datas)):  # 去重
            self.train_datas[i] = set(self.train_datas[i])
        self.k_train_datas, self.k_train_labels, self.k_validation_datas, self.k_validation_labels = \
        io_data.k_fold_cross_validation(self.train_datas, self.train_labels)

    # 读取测试集
    def read_test(self, test_data_path):
        self.test_datas = io_data.read_test(test_data_path)
        for i in range(len(self.test_datas)):  # 去重
            self.test_datas[i] = set(self.test_datas[i])

    # 闵科夫斯基距离
    def calculate_lp(self, text1, text2):
        return len( text1 ^ text2 )

    # 余弦距离取负
    def calculate_cos(self, text1, text2):
        return - len(text1 & text2) / (len(text1) + len(text2))

    # 杰卡德距离取负
    def calculate_jaccard(self, text1, text2):
        return - len(text1 & text2) / len(text1 | text2)

    # KNN最近邻算法
    def k_nearest_neighbors(self, train_datas, train_labels, validation_datas, k):
        '''
        :param train_datas: 训练集数据
        :param train_labels: 训练集标签
        :param validation_datas: 验证集数据
        :param k: 最近邻的邻居个数
        :return: 预测的验证集标签
        '''
        predict_labels = []
        for i in range(len(validation_datas)):
            ndistance = []
            for j in range(len(train_datas)):
                distance = self.calculate_jaccard(validation_datas[i], train_datas[j])
                ndistance.append([distance, train_labels[j]])
            knearest = [distance[1] for distance in heapq.nsmallest(k, ndistance)]
            predict_labels.append(collections.Counter(knearest).most_common(1)[0][0])
        return predict_labels

    # 交叉验证
    def cross_validation(self, k):
        '''
        :param k: 最近邻的邻居个数K
        :return: 十折交叉验证的平均正确率
        '''
        average_accuracy = 0.0
        for i in range(10):
            predict_labels = self.k_nearest_neighbors(self.k_train_datas[i], self.k_train_labels[i], self.k_validation_datas[i], k)
            predict_accuracy = (np.array(predict_labels)==np.array(self.k_validation_labels[i])).tolist().count(True) / len(predict_labels)
            average_accuracy += predict_accuracy
        average_accuracy /= 10
        return average_accuracy

    # 遍历寻找最佳的K
    def traversal_k(self):
        accuracies = []
        for k in range(1, 200, 10):
            accuracy = self.cross_validation(k)
            print("The accuracy of K=" + str(k) + " is " + str(accuracy))
            accuracies.append(accuracy)
        print("The best average accuracy is " + str(max(accuracies)))
        print("The best parameter K is " + str(accuracies.index(max(accuracies))))

    # 写入测试集
    def write_test(self, test_label_path):
        predict_labels = self.k_nearest_neighbors(self.train_datas, self.train_labels, self.test_datas, 10)
        io_data.write_test(test_label_path, predict_labels)

if __name__ == '__main__':
    knn = KNN()
    knn.read_train("data/2/trainData.txt", "data/2/trainLabel.txt")
    knn.read_test("data/2/testData.txt")
    knn.write_test("data/2/16337250_1.txt")
