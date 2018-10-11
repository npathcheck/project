import tensorflow as tf
import numpy as np
import time


train_datas = []
train_labels = []
test_datas = []
embedding_size = 256
filter_sizes = [3, 4, 5]
num_filters = 1024
batch_size = 64
num_epochs = 10



read_object = open("data/2/unclearTrainData.txt", 'r', encoding='UTF-8')
for line in read_object.readlines():
    train_datas.append(line.strip())
read_object.close()
read_object = open("data/2/unclearTestData.txt", 'r', encoding='UTF-8')
for line in read_object.readlines():
    test_datas.append(line.strip())
read_object.close()
read_object = open("data/2/trainLabel.txt", 'r', encoding='UTF-8')
for line in read_object.readlines():
    train_labels.append(int(line.strip()))
read_object.close()

length = 500
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(length)
datas = np.array(list(vocab_processor.fit_transform(train_datas+test_datas)))
vocab_size = len(vocab_processor.vocabulary_)
train_datas = datas[:24000]
test_datas = datas[24000:]
train_labels = np.c_[1-np.array(train_labels), np.array(train_labels)]

input_x = tf.placeholder(tf.int32, [None, length], name="input_x")
input_y = tf.placeholder(tf.float32, [None, train_labels.shape[1]], name="input_y")
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
with tf.device('/cpu:0'), tf.name_scope("embedding"):
    Weights = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="Weights")
    # shape:[None, sequence_length, embedding_size]
    embedded_chars = tf.nn.embedding_lookup(Weights, input_x)
    # 添加一个维度，shape:[None, sequence_length, embedding_size, 1]
    embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

pooled_outputs = []
for i, filter_size in enumerate(filter_sizes):
    with tf.name_scope("conv-maxpool-%s" % filter_size):
        # filter卷积核高度，embedding_size卷积核宽度，通道数，num_filter卷积核数
        filter_shape = [filter_size, embedding_size, 1, num_filters]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
        conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        # h需要池化的输入，ksize池化窗口的大小，[1，height,width,1]
        pooled = tf.nn.max_pool(h, ksize=[1, length  - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
        pooled_outputs.append(pooled)

def batch_iter(data, batch_size, num_epochs):
    data = np.array(data)
    data_size = len(data)
    # 每个epoch的num_batch
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield data[start_index:end_index]

num_filters_total = num_filters * len(filter_sizes)
h_pool = tf.concat(pooled_outputs, 3)
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

with tf.name_scope("dropout"):
    h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

with tf.name_scope("output"):
    W = tf.get_variable("W", shape=[num_filters_total, 2], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
    scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
    predictions = tf.argmax(scores, 1, name="predictions")

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=scores, labels=input_y))

with tf.name_scope("accuracy"):
    correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=1)
if __name__ == "__main__":
    with tf.Session() as sess:
        sess.run(init)
        #saver.restore(sess, "check_point/2/unclear/")
        batches = batch_iter(list(zip(train_datas, train_labels)), batch_size, num_epochs)
        i = 0
        for batch in batches:
            batch_datas, batch_labels = zip(*batch)
            _loss, _accuracy, _optimizer = sess.run([loss, accuracy, optimizer], feed_dict={input_x: batch_datas, input_y: batch_labels, dropout_keep_prob: 0.5})
            print(str(i) + "th loss {:g}, acc {:g}".format(_loss, _accuracy))
            i += 1
            if i % 375 == 0:
                save_path = saver.save(sess, "check_point/2/unclear/")
                print("save model")
                time.sleep(120)
        predict_labels = []
        for i in range(60):
            test_data = test_datas[i*100:(i+1)*100]
            predict_labels += list(sess.run([predictions], feed_dict={input_x: test_data, dropout_keep_prob: 1})[0])
        write_object = open("data/2/16337250_0.txt", 'w', encoding='UTF-8')
        for predict_label in predict_labels:
            write_object.write(str(predict_label) + '\n')
        write_object.close()



