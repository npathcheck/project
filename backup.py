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


# import IO_regression
# import numpy as np
# import datetime
# #
# class NN():
#
#     learning_rate = 0.0001
#
#     train_datas = []
#     train_tags = []
#     train_labels = []
#     test_datas = []
#     test_tags = []
#     predict_labels = []
#
#     def read(self, file_path):
#         self.train_datas, self.train_tags, self.train_labels = IO_regression.read(file_path, "train")
#         self.test_datas, self.test_tags = IO_regression.read(file_path, "test")
#         self.zero_one_standard(self.train_datas)
#         self.zero_one_standard(self.test_datas)
#
#     def write(self, file_path):
#         IO_regression.write(file_path, self.predict_labels)
#
#     def zero_one_standard(self, data):
#         for i in range(6):
#             data[:,i] = (data[:,i] - np.min(data[:,i])) / (np.max(data[:,i] - np.min(data[:,i]))) * 10
#
#     def initial(self):
#         self.weights = np.random.randn(6, 1) / np.sqrt(6)
#         self.biases = 0
#
#     def neuron_network(self):
#         loss = []
#         for i in range(len(self.train_datas)):
#             predict_y = np.dot(self.weights.T, self.train_datas[i]) + self.biases
#             loss.append((predict_y - self.train_labels[i]) ** 2 / 2)
#             self.weights = self.weights - (self.learning_rate * (predict_y - self.train_labels[i]) * self.train_datas[i]).reshape(6, 1)
#             self.biases = self.biases - self.learning_rate * (predict_y - self.train_labels[i])
#         return np.mean(np.array(loss))
#
#     def calculate_result(self):
#         for i in range(len(self.test_datas)):
#             predict_y = np.dot(self.weights.T, self.train_datas[i]) + self.biases
#             self.predict_labels.append(predict_y[0])
#
# if __name__ == "__main__":
#     np.random.seed()
#     nn = NN()
#     nn.read("data/回归")
#     nn.initial()
#     for i in range(200):
#         print(str(i) + " epoch: " + str(nn.neuron_network()))
#     nn.calculate_result()
#     nn.write("data/回归/16337250_2.txt")


# import IO_regression
# import numpy as np
# import datetime
#
# class NN():
#
#     learning_rate = 0.0001
#
#     train_datas = []
#     train_tags = []
#     train_labels = []
#     test_datas = []
#     test_tags = []
#     predict_labels = []
#
#     def read(self, file_path):
#         self.train_datas, self.train_tags, self.train_labels = IO_regression.read(file_path, "train")
#         self.test_datas, self.test_tags = IO_regression.read(file_path, "test")
#         self.zero_one_standard(self.train_datas)
#         self.zero_one_standard(self.test_datas)
#
#     def write(self, file_path):
#         IO_regression.write(file_path, self.predict_labels)
#
#     def zero_one_standard(self, data):
#         for i in range(6):
#             data[:,i] = (data[:,i] - np.min(data[:,i])) / (np.max(data[:,i] - np.min(data[:,i]))) * 10
#
#     def initial(self):
#         self.weights = [np.random.randn(6, 6) / np.sqrt(6), np.random.randn(6, 1) / np.sqrt(6)]
#         self.biases = [np.zeros(6), 0]
#
#     def sigmoid(self, x):
#         return 1 / (1+np.exp(-x))
#
#     def sigmoid_derivative(self, x):
#         return x * (1-x)
#
#     def neuron_network(self):
#         loss = []
#         for i in range(len(self.train_datas)):
#             hidden_y = self.sigmoid(np.dot(self.weights[0].T, self.train_datas[i]) + self.biases[0])
#             output_y = np.dot(self.weights[1].T, hidden_y) + self.biases[1]
#             error = output_y[0] - self.train_labels[i]
#             loss.append(error ** 2 / 2)
#             self.weights[1] = self.weights[1] - (self.learning_rate * error * hidden_y).reshape(6,1)
#             self.biases[1] = self.biases[1] - (self.learning_rate * error)
#             self.weights[0] = self.weights[0] - (self.learning_rate * error * hidden_y * (1-hidden_y) * np.dot(self.weights[1], self.train_datas[i].reshape(1, 6)).T)
#             self.biases[0] = self.biases[0] - (self.learning_rate * error * hidden_y * (1 - hidden_y) * self.weights[1].reshape(6))
#         return np.mean(np.array(loss))
#
#     def calculate_result(self):
#         for i in range(len(self.test_datas)):
#             hidden_y = self.sigmoid(np.dot(self.weights[0].T, self.train_datas[i]) + self.biases[0])
#             output_y = np.dot(self.weights[1].T, hidden_y) + self.biases[1]
#             self.predict_labels.append(output_y[0])
#
# if __name__ == "__main__":
#     np.random.seed()
#     nn = NN()
#     nn.read("data/回归")
#     nn.initial()
#     nn.neuron_network()
#     for i in range(200):
#         print(str(i) + " epoch: " + str(nn.neuron_network()))
#     nn.calculate_result()
#     nn.write("data/回归/16337250_3.txt")

import tensorflow as tf
import numpy as np


train_datas = []
train_labels = []
test_datas = []
read_object = open("data/2/unclearPaddingTrainData.txt", 'r', encoding='UTF-8')
for line in read_object.readlines():
    train_datas.append(line.strip())
read_object.close()
read_object = open("data/2/trainLabel.txt", 'r', encoding='UTF-8')
for line in read_object.readlines():
    train_labels.append(int(line.strip()))
read_object.close()
read_object = open("data/2/unclearPaddingTestData.txt", 'r', encoding='UTF-8')
for line in read_object.readlines():
    test_datas.append(line.strip())
read_object.close()

max_length = len(train_datas[0].split())
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_length)
datas = np.array(list(vocab_processor.fit_transform(train_datas + test_datas)))
vocab_processor.save("vocab_dict")
train_datas = datas[:24000]
test_datas = datas[24000:]
train_labels_temp = np.zeros([len(train_labels), 2])
for i in range(len(train_labels)):
    train_labels_temp[i, train_labels[i]] = 1
train_labels = train_labels_temp

input_x = tf.placeholder(tf.int32, [None, train_datas.shape[1]], name="input_x")
input_y = tf.placeholder(tf.float32, [None, train_labels.shape[1]], name="input_y")

dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

num_classes = 2                                     # num_classes-分类数
vocab_size=len(vocab_processor.vocabulary_)         # vocab_size-总词汇数
embedding_size = 256                                # embedding_size-词向量长度
filter_sizes= [3, 4, 5]                             # filter_sizes-卷积核尺寸3，4，5
num_filters = 1024                                  # num_filters-卷积核数量
batch_size = 64
num_epochs = 10
evaluate_every = 50

Weights = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="Weights")
# shape:[None, sequence_length, embedding_size]
embedded_chars = tf.nn.embedding_lookup(Weights, input_x)
# 添加一个维度，shape:[None, sequence_length, embedding_size, 1]
embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

# 对于每种卷积尺寸构造卷积层和池化层
pooled_outputs = []
for i, filter_size in enumerate(filter_sizes):
    with tf.name_scope("conv-maxpool-%s" % filter_size):
        # 卷积层
        filter_shape = [filter_size, embedding_size, 1, num_filters]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
        conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
        # relu激活函数
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        # 池化层
        pooled = tf.nn.max_pool(h, ksize=[1, max_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
        pooled_outputs.append(pooled)

# 合成大的特征向量
num_filters_total = num_filters * len(filter_sizes)
print("num_filters_total:", num_filters_total)
h_pool = tf.concat(pooled_outputs, 3)
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

# 增加dropout
with tf.name_scope("dropout"):h_drop = tf.nn.dropout(h_pool_flat,dropout_keep_prob)

# 最后没有归一化的分数
with tf.name_scope("output"):
    W = tf.get_variable("W", shape=[num_filters_total, num_classes], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
    scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")

# 定义loss
with tf.name_scope("loss"):
    print(scores.shape)
    print(input_y.shape)
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=scores, labels=input_y))

# 定义优化器
with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)

# 生成批次数据
def batch_iter(data, batch_size, num_epochs, shuffle=False):
    data = np.array(data)
    data_size = len(data)
    # 每个epoch的num_batch
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    print("num_batches_per_epoch:",num_batches_per_epoch)
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

with tf.Session() as sess:
    predict_top_2 = tf.nn.top_k(scores, k=2)
    label_top_2 = tf.nn.top_k(input_y, k=2)
    sess.run(tf.global_variables_initializer())
    i = 0
    # 生成数据
    batches = batch_iter(list(zip(train_datas, train_labels)), batch_size, num_epochs)
    for batch in batches:
        i = i + 1
        # 得到一个batch的数据
        x_batch, y_batch = zip(*batch)
        # 优化模型
        sess.run([optimizer],feed_dict={input_x:x_batch, input_y:y_batch, dropout_keep_prob:dropout_keep_prob})
        # 每训练50次测试1次
        if (i % evaluate_every == 0):
            print ("Evaluation:step",i)
            predict_2, label_2, _loss = sess.run([predict_top_2, label_top_2, loss],feed_dict={input_x:x_batch, input_y:y_batch, dropout_keep_prob:1.0})
            print ("label:",label_2[1][:2])
            print ("predict:",predict_2[1][:2])
            print ("predict:",predict_2[0][:2])
            print ("loss:",_loss)
            predict_label_and_marked_label_list = []
            for predict, label in zip(predict_2[1],label_2[1]):
                predict_label_and_marked_label_list.append((list(predict),list(label)))
