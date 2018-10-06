import math
import numpy as np
import IO_classification

class NB():

    train_datas = []
    train_labels = []
    test_datas = []
    predict_labels = []
    wordList = []
    wordDict = {}
    wordMatrix= []

    # 读取训练集
    def read(self, file_path):
        self.train_datas, self.train_labels, self.test_datas = IO_classification.read(file_path)

    def write(self, file_path):
        IO_classification.write(file_path, self.predict_labels)

    # 创建词向量矩阵
    def create_wordMatrix(self, train_datas, train_labels):
        self.wordList = set()
        for train_data in train_datas:
            self.wordList |= set(train_data)
        self.wordList = list(self.wordList)
        for i in range(len(self.wordList)):
            self.wordDict[self.wordList[i]] = i
        words_matrix = np.zeros([len(set(train_labels)), len(self.wordList)])
        for i in range(len(train_datas)):
            for j in range(len(train_datas[i])):
                words_matrix[train_labels[i], self.wordDict[train_datas[i][j]]] += 1
        words_amount = np.sum(words_matrix, axis=1)
        # 最后一列用于存储训练集中不存在的单词的词向量
        self.wordMatrix = np.zeros([len(set(train_labels)), len(self.wordList)+1])
        wordLen = len(self.wordList)
        for i in range(self.wordMatrix.shape[0]):
            for j in range(self.wordMatrix.shape[1]-1):
                self.wordMatrix[i,j] = math.log((words_matrix[i,j]+1)/(words_amount[i]+wordLen))
            self.wordMatrix[i, -1] = math.log(1/(words_amount[i]+wordLen+1))

    # 朴素贝叶斯
    def navie_bayes(self, train_labels, validation_datas):
        predict_labels = []
        step = 0
        for validation_data in validation_datas:
            print("step: " + str(step))
            step += 1
            labels_possibility = [0] * self.wordMatrix.shape[0]
            for label in range(self.wordMatrix.shape[0]):
                labels_possibility[label] += math.log(train_labels.count(label) / len(train_labels))
                for word in validation_data:
                    if word in self.wordList:
                        labels_possibility[label] += self.wordMatrix[label, self.wordDict[word]]
                    else:
                        labels_possibility[label] += self.wordMatrix[label, -1]
            predict_labels.append(labels_possibility.index(max(labels_possibility)))
        return predict_labels

    # 交叉验证
    def cross_validation(self):
        k_train_datas, k_train_labels, k_validation_datas, k_validation_labels = IO_classification.k_fold_cross_validation(self.train_datas, self.train_labels)
        average_accuracy = 0.0
        for i in range(10):
            self.create_wordMatrix(k_train_datas[i], k_train_labels[i])
            predict_labels = self.navie_bayes(k_train_labels[i], k_validation_datas[i])
            predict_accuracy = (np.array(predict_labels) == np.array(k_validation_labels[i])).tolist().count(True) / len(predict_labels)
            average_accuracy += predict_accuracy
            print("The accuracy of " + str(i+1) + "th cross validation is " + str(predict_accuracy))
        average_accuracy /= 10
        print("The average accuracy of cross validation is " + str(average_accuracy))
        return average_accuracy


if __name__ == '__main__':
    nb = NB()
    nb.read("data/5/")
    nb.create_wordMatrix(nb.train_datas, nb.train_labels)
    nb.predict_labels = nb.navie_bayes(nb.train_labels, nb.test_datas)
    nb.write("data/5/")




