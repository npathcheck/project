import collections
import input_data
import numpy as np


class KNN():

    k_train_datas = []
    k_train_labels = []
    k_validation_datas = []
    k_validation_labels = []

    # 读取训练集
    def read_train(self, train_data_path, train_label_path):
        train_datas, train_lables = input_data.read_train(train_data_path, train_label_path)
        for i in range(len(train_datas)):  # 去重
            train_datas[i] = set(train_datas[i])
        self.k_train_datas, self.k_train_labels, self.k_validation_datas, self.k_validation_labels = \
        input_data.k_fold_cross_validation(train_datas, train_lables)

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
            knearest = [distance[1] for distance in sorted(ndistance)][:k]
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
        for k in range(1, 102, 10):
            accuracy = self.cross_validation(k)
            print("The accuracy of K=" + str(k) + " is " + str(accuracy))
            accuracies.append(accuracy)
        print("The best average accuracy is " + str(max(accuracies)))
        print("The best parameter K is " + str(accuracies.index(max(accuracies))))

if __name__ == '__main__':
    knn = KNN()
    knn.read_train("data/2/trainData.txt", "data/2/trainLabel.txt")
    knn.traversal_k()
