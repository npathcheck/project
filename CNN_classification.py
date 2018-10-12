import tensorflow as tf
import numpy as np
import time

class CNN():

    train_datas = []
    train_labels = []
    test_datas = []
    predict_labels = []
    embedding_size = 256
    filter_sizes = [3, 4, 5]
    num_filters = 1024
    batch_size = 64
    num_epochs = 1
    sentence_length = 500


    def read(self):
        read_object = open("data/5/unclearTrainData.txt", 'r', encoding='UTF-8')
        for line in read_object.readlines():
            self.train_datas.append(line.strip())
        read_object.close()
        read_object = open("data/5/unclearTestData.txt", 'r', encoding='UTF-8')
        for line in read_object.readlines():
            self.test_datas.append(line.strip())
        read_object.close()
        read_object = open("data/5/trainLabel.txt", 'r', encoding='UTF-8')
        for line in read_object.readlines():
            self.train_labels.append(int(line.strip()))
        read_object.close()

    def write(self, file_path):
        write_object = open(file_path, 'w', encoding='UTF-8')
        for predict_label in self.predict_labels:
            write_object.write(str(predict_label) + '\n')
        write_object.close()

    def create_vocab_processor(self):
        vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(self.sentence_length)
        datas = np.array(list(vocab_processor.fit_transform(self.train_datas+self.test_datas)))
        self.vocab_size = len(vocab_processor.vocabulary_)
        self.train_datas = datas[:24000]
        self.test_datas = datas[24000:]
        self.train_labels_array = np.zeros([len(self.train_labels), 5])
        for i in range(len(self.train_labels)):
            self.train_labels_array[i,self.train_labels[i]] = 1
        self.train_labels =self.train_labels_array

    def create_model(self):

        self.input_x = tf.placeholder(tf.int32, [None, self.sentence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.train_labels.shape[1]], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embedded_weights = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), name="Weights")
            embedded_chars = tf.nn.embedding_lookup(embedded_weights, self.input_x)
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("convolution"):
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool(h, ksize=[1, self.sentence_length  - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pooled)

        num_filters_total = self.num_filters * len(self.filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)

        with tf.name_scope("output"):
            W = tf.get_variable("W", shape=[num_filters_total, 5], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[5]), name="b")
            self.scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.scores, labels=self.input_y))

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss)

    def train(self):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=1)
        with tf.Session() as sess:
            sess.run(init)
            #saver.restore(sess, "check_point/5/cnn_unclear/")
            batches = batch_iter(list(zip(self.train_datas, self.train_labels)), self.batch_size, self.num_epochs)
            i = 0
            for batch in batches:
                batch_datas, batch_labels = zip(*batch)
                _loss, _accuracy, _optimizer = sess.run([self.loss, self.accuracy, self.optimizer],
                                                        feed_dict={self.input_x: batch_datas, self.input_y: batch_labels, self.dropout_keep_prob: 0.5})
                print(str(i) + "th loss {:g}, acc {:g}".format(_loss, _accuracy))
                i += 1
                if i % 375 == 0:
                    saver.save(sess, "check_point/5/cnn_unclear/")
                    print("save model")
                if i % 3750 == 0:
                    self.predict_labels = []
                    for j in range(60):
                        test_data = self.test_datas[j * 100:(j + 1) * 100]
                        self.predict_labels += list(sess.run([self.predictions], feed_dict={self.input_x: test_data, self.dropout_keep_prob: 1})[0])
                    self.write("data/5/" + str(i) + ".txt")

def batch_iter(data, batch_size, num_epochs):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield data[start_index:end_index]


if __name__ == "__main__":
    cnn = CNN()
    cnn.read()
    cnn.create_vocab_processor()
    cnn.create_model()
    cnn.train()
