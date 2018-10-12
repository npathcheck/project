import tensorflow as tf
import numpy as np
import time

class RNN():

    train_datas = []
    train_labels = []
    test_datas = []
    predict_labels = []
    batch_size = 64
    time_step = 64
    num_epochs = 50
    embedding_size = 256
    sentence_length = 500
    neuron_num = 128
    layer_num = 4
    class_num = 2
    learning_rate = 1e-4

    def read(self):
        read_object = open("data/2/unclearTrainData.txt", 'r', encoding='UTF-8')
        for line in read_object.readlines():
            self.train_datas.append(line.strip())
        read_object.close()
        read_object = open("data/2/unclearTestData.txt", 'r', encoding='UTF-8')
        for line in read_object.readlines():
            self.test_datas.append(line.strip())
        read_object.close()
        read_object = open("data/2/trainLabel.txt", 'r', encoding='UTF-8')
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
        self.train_labels = np.c_[1 - np.array(self.train_labels), np.array(self.train_labels)]

    def create_basic_lstm_cell(self):
        basic_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.neuron_num)
        basic_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(basic_lstm_cell, output_keep_prob=self.dropout_keep_prob)
        return basic_lstm_cell

    def create_multi_lstm_cell(self):
        multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([self.create_basic_lstm_cell() for _ in range(self.layer_num)])
        initial_state = multi_lstm_cell.zero_state(self.sentence_length, dtype=tf.float32)
        return multi_lstm_cell, initial_state

    def create_weights_variable(self, shape, name="weights"):
        initializer = tf.random_normal_initializer(mean=0., stddev=0.5)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def create_biases_variable(self, shape, name="biases"):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def create_model(self):

        with tf.name_scope("placeholder"):
            self.input_x = tf.placeholder(tf.int64, [self.batch_size, self.sentence_length], name="input_x")
            self.input_y = tf.placeholder(tf.int64, [self.batch_size, self.train_labels.shape[1]], name="input_y")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embedded_weights = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), name="Weights")
            # embedded_chars.shape = (batch_size, sequence_length, embedding_size)
            embedded_chars = tf.nn.embedding_lookup(embedded_weights, self.input_x)
            # dropout_x.shape = (batch_size, sequence_length, embedding_size)
            dropout_x = tf.nn.dropout(embedded_chars, self.dropout_keep_prob)

        with tf.name_scope("rnn"):
            with tf.variable_scope("input"):
                input_x = tf.reshape(dropout_x, [-1, self.embedding_size])
                input_weights = self.create_weights_variable([self.embedding_size ,self.neuron_num], name="input_weights")
                input_biases = self.create_biases_variable([self.neuron_num], name="input_biases")
                input_y = tf.nn.xw_plus_b(input_x, input_weights, input_biases)
                # input_y.shape = (sentence_size, time_step, neuron_num)
                input_y = tf.reshape(input_y, [-1, self.time_step, self.neuron_num])
            with tf.variable_scope("lstm_cell"):
                # lstm_outputs.shape = (batch_size, sentence_size, neuron_num)
                lstm_outputs = []
                lstm_cell, init_state = self.create_multi_lstm_cell()
                # input_y.shape = (sentence_size, neuron_num)
                # init_state.shape = (layer_num, class_num)
                lstm_output, state = lstm_cell(input_y[:, 0, :], init_state)
                lstm_outputs.append(lstm_output)
                for i in range(1, self.time_step):
                    tf.get_variable_scope().reuse_variables()
                    lstm_output, state = lstm_cell(input_y[:, i, :], init_state)
                    lstm_outputs.append(lstm_output)
            with tf.variable_scope("output"):
                output_x = tf.reduce_mean(lstm_outputs ,1)
                output_weights = self.create_weights_variable([self.neuron_num, self.class_num], name="output_weights")
                output_biases = self.create_biases_variable([self.class_num], name="output_biases")
                output_y = tf.nn.xw_plus_b(output_x, output_weights, output_biases)

        with tf.name_scope("loss"):
            # +1e-10 防止output_y为0导致softmax_cross_entropy_with_logits输出nan
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_y + 1e-10, labels=self.input_y), name="loss")

        with tf.name_scope("accuracy"):
            self.prediction = tf.argmax(output_y, 1)
            correct_prediction = tf.equal(self.prediction, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def train(self):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=1)
        with tf.Session() as sess:
            sess.run(init)
            #saver.restore(sess, "check_point/2/rnn_unclear/")
            batches = batch_iter(list(zip(self.train_datas, self.train_labels)), self.batch_size, self.num_epochs)
            i = 0
            for batch in batches:
                batch_datas, batch_labels = zip(*batch)
                _loss, _accuracy, _optimizer = sess.run([self.loss, self.accuracy, self.optimizer],
                                                        feed_dict={self.input_x: batch_datas, self.input_y: batch_labels, self.dropout_keep_prob: 0.5})
                print(str(i) + "th loss {:g}, acc {:g}".format(_loss, _accuracy))
                i += 1
                if i % 375 == 0:
                    saver.save(sess, "check_point/2/rnn_unclear/")
                    print("save model")
                    time.sleep(300)
                if i % 3750 == 0:
                    self.predict_labels = []
                    for j in range(60):
                        test_data = self.test_datas[j * 100:(j + 1) * 100]
                        self.predict_labels += list(sess.run([self.prediction], feed_dict={self.input_x: test_data, self.dropout_keep_prob: 1})[0])
                    self.write("data/2/" + str(i) + ".txt")


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
    rnn = RNN()
    rnn.read()
    rnn.create_vocab_processor()
    rnn.create_model()
    rnn.train()









