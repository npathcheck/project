import tensorflow as tf
import numpy as np
import time

class RNN():

    train_datas = []
    train_labels = []
    test_datas = []
    predict_labels = []
    batch_size = 64
    num_epochs = 50
    embedding_size = 256
    sentence_length = 500
    keep_prob = 0.5
    neuron_num = 256
    layer_num = 3

    def create_vocab_processor(self):
        vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(self.sentence_length)
        datas = np.array(list(vocab_processor.fit_transform(self.train_datas+self.test_datas)))
        self.vocab_size = len(vocab_processor.vocabulary_)
        self.train_datas = datas[:24000]
        self.test_datas = datas[24000:]
        self.train_labels = np.c_[1-np.array(self.train_labels), np.array(self.train_labels)]

    def get_basic_lstm_cell(self):
        basic_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.neuron_num)
        basic_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(basic_lstm_cell, output_keep_prob=self.keep_prob)
        return basic_lstm_cell

    def create_multi_lstm_cell(self):
        self.multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([self.get_basic_lstm_cell() for _ in range(self.layer_num)])
        self.initial_state = self.multi_lstm_cell.zero_state(self.batch_size, dtype=tf.float32)

    def create_model(self):

        self.input_x = tf.placeholder(tf.int32, [None, self.sentence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.train_labels.shape[1]], name="input_y")

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embedded_weights = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), name="Weights")
            embedded_chars = tf.nn.embedding_lookup(embedded_weights, self.input_x)

        with tf.name_scope("dropout"):
            input = tf.nn.dropout(embedded_chars, self.keep_prob)

        lstm_outputs = []
        with tf.name_scope("lstm"):
            for time_step in range(self.sentence_length):
                if time_step != 0:                                  # 重复利用RNN的参数
                    tf.get_variable_scope().reuse_variables()





