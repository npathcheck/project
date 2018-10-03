import collections
import input_data


class KNN():

    k = 10
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
        input_data.k_fold_cross_validation(train_datas, train_lables, self.k)

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
    def k_nearest_neighbors(self, train_datas, train_labels, validation_datas, k=10):
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
    def cross_validation(self):
        for i in range(self.k):
            predict_labels = self.k_nearest_neighbors(self.k_train_datas[i], self.k_train_labels[i], self.k_validation_datas[i])
            predict_accuracy = [predict_labels == self.k_validation_labels[i]].count(True) / len(predict_labels)
            print("The accuracy of " + str(i + 1) + "th K-fold cross validation is " + str(predict_accuracy))


if __name__ == '__main__':
    knn = KNN()
    knn.read_train("data/2/trainData.txt", "data/2/trainLabel.txt")
    knn.cross_validation()
