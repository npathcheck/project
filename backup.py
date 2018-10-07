# import numpy as np
# import IO_classification
# import tensorflow as tf
# from tensorflow.contrib import learn
#
# class CNN():
#
#     train_datas = []
#     train_labels = []
#     test_datas = []
#     predict_labels = []
#     learning_rage = 1e-3
#
#     def read(self, file_path):
#         self.train_datas, self.train_labels, self.test_datas = IO_classification.read(file_path, "text")
#         datas = self.train_datas + self.test_datas
#         self.sequence_length = max([len(data) for data in datas])
#         self.classes_amount = len(set(self.train_labels))
#         self.vocab_processor = learn.preprocessing.VocabularyProcessor(self.sequence_length)
#         datas = np.array(list(self.vocab_processor.fit_transform(datas)))
#         self.train_datas = datas[:24000]
#         self.test_datas = datas[24000:]
#
#     def convolution_neural_network(self):
#         self.input_x = tf.placeholder(tf.int32, [None, self.sequence_length], name="input_x")
#         self.input_y = tf.placeholder(tf.float32, [None, self.classes_amount], name="input_y")
#         self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
#         l2_loss = tf.constant(0.0)
#         with tf.device('/cpu:0'):
#             self.W = tf.Variable(tf.random_uniform([len(self.vocab_processor.vocabulary_), 128], -1.0, 1.0),name="W")
#             self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
#             self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
#
#
#
# if __name__ == '__main__':
#     cnn = CNN()
#     cnn.read("data/2")
#     # 未完成


import IO_regression
import numpy as np
import datetime
#
class NN():

    learning_rate = 0.0001

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

    def initial(self):
        self.weights = np.random.randn(6, 1) / np.sqrt(6)
        self.biases = 0

    def neuron_network(self):
        loss = []
        for i in range(len(self.train_datas)):
            predict_y = np.dot(self.weights.T, self.train_datas[i]) + self.biases
            loss.append((predict_y - self.train_labels[i]) ** 2 / 2)
            self.weights = self.weights - (self.learning_rate * (predict_y - self.train_labels[i]) * self.train_datas[i]).reshape(6, 1)
            self.biases = self.biases - self.learning_rate * (predict_y - self.train_labels[i])
        return np.mean(np.array(loss))

    def calculate_result(self):
        for i in range(len(self.test_datas)):
            predict_y = np.dot(self.weights.T, self.train_datas[i]) + self.biases
            self.predict_labels.append(predict_y[0])

if __name__ == "__main__":
    np.random.seed()
    nn = NN()
    nn.read("data/回归")
    nn.initial()
    for i in range(200):
        print(str(i) + " epoch: " + str(nn.neuron_network()))
    nn.calculate_result()
    nn.write("data/回归/16337250_2.txt")
