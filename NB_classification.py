import math
import numpy as np
import input_data

class NB():

    k_train_datas = []
    k_train_labels = []
    k_validation_datas = []
    k_validation_labels = []
    wordList = []
    word2vec= []

    # 读取训练集
    def read_train(self, train_data_path, train_label_path):
        train_datas, train_lables = input_data.read_train(train_data_path, train_label_path)
        self.k_train_datas, self.k_train_labels, self.k_validation_datas, self.k_validation_labels = \
            input_data.k_fold_cross_validation(train_datas, train_lables)

    # 创建词向量矩阵
    def create_word2vec(self, train_datas, train_lables):
        self.wordList = set()
        for train_data in train_datas:
            self.wordList |= set(train_data)
        self.wordList = list(self.wordList)
        words_matrix = np.zeros([len(set(train_lables)), len(self.wordList)])
        for i in range(len(train_datas)):
            for j in range(len(train_datas[i])):
                words_matrix[train_lables[i], self.wordList.index(train_datas[i][j])] += 1
        words_amount = np.sum(words_matrix, axis=1)
        # 最后一列用于存储训练集中不存在的单词的词向量
        self.word2vec = np.zeros([len(set(train_lables)), len(self.wordList)+1])
        for i in range(self.word2vec.shape[0]):
            for j in range(self.word2vec.shape[1]-1):
                self.word2vec[i,j] = math.log((words_matrix[i,j]+1)/(words_amount[i]+len(self.wordList)))
            self.word2vec[i, -1] = math.log(1/(words_amount[i]+len(self.wordList))*len(self.wordList))

    # 朴素贝叶斯
    def navie_bayes(self, train_labels, validation_datas):
        predict_labels = []
        for validation_data in validation_datas:
            lables_possibility = [0] * self.word2vec.shape[0]
            for label in range(self.word2vec.shape[0]):
                lables_possibility[label] += math.log(train_labels.count(label) / len(train_labels))
                for word in validation_data:
                    if word in self.wordList:
                        lables_possibility[label] += self.word2vec[label, self.wordList.index(word)]
                    else:
                        lables_possibility[label] += self.word2vec[label, -1]
            predict_labels.append(lables_possibility.index(max(lables_possibility)))
        return predict_labels

    # 交叉验证
    def cross_validation(self):
        average_accuracy = 0.0
        for i in range(10):
            self.create_word2vec(self.k_train_datas[i], self.k_train_labels[i])
            predict_labels = self.navie_bayes(self.k_train_labels[i], self.k_validation_datas[i])
            predict_accuracy = (np.array(predict_labels) == np.array(self.k_validation_labels[i])).tolist().count(True) / len(predict_labels)
            average_accuracy += predict_accuracy
            print("The accuracy of " + str(i+1) + "th cross validation is " + str(predict_accuracy))
        average_accuracy /= 10
        print("The average accuracy of cross validation is " + str(average_accuracy))
        return average_accuracy

if __name__ == '__main__':
    nb = NB()
    nb.read_train("data/2/trainData.txt", "data/2/trainLabel.txt")
    nb.cross_validation()

