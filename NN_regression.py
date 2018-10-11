import IO_regression
import numpy as np

class NN():

    learning_rate = 0.0001
    dropout_prob = 0.5
    neuron_unit = 40
    input_size = 20

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
        tags = np.load(file_path + "/tag_word2vec.npz")
        self.train_tags_array = tags["arr_0"]
        self.test_tags_array = tags["arr_1"]
        self.train_datas = np.c_[self.train_datas, self.train_tags_array]
        self.test_datas = np.c_[self.test_datas, self.test_tags_array]
        # self.train_datas = tags["arr_0"]
        # self.test_datas = tags["arr_1"]
        # self.input_size = self.train_datas.shape[1]

    def write(self, file_path):
        IO_regression.write(file_path, self.predict_labels)

    def zero_one_standard(self, data):
        for i in range(6):
            data[:,i] = (data[:,i] - np.min(data[:,i])) / (np.max(data[:,i] - np.min(data[:,i]))) * 10

    def initial(self):
        self.weights = [np.random.randn(self.input_size, self.neuron_unit) / np.sqrt(self.input_size),
                        np.random.randn(self.neuron_unit, self.neuron_unit) / np.sqrt(self.neuron_unit),
                        np.random.randn(self.neuron_unit, 1) / np.sqrt(self.neuron_unit)]
        self.biases = [np.zeros(self.neuron_unit), np.zeros(self.neuron_unit), np.array(0)]

    def sigmoid(self, x):
        return 1 / (1+np.exp(-x))

    def sigmoid_neuron_network(self):
        loss = []
        for i in range(len(self.train_datas)):
            hidden1_y = self.sigmoid(np.dot(self.weights[0].T, self.train_datas[i]) + self.biases[0])
            hidden2_y = self.sigmoid(np.dot(self.weights[1].T, hidden1_y) + self.biases[1])
            output_y = np.dot(self.weights[2].T, hidden2_y) + self.biases[2]
            error = output_y[0] - self.train_labels[i]
            loss.append(error ** 2 / 2)

            delta = self.learning_rate * error
            self.weights[2] = self.weights[2] - (delta * hidden2_y).reshape(self.neuron_unit, 1)
            self.biases[2] = self.biases[2] - delta

            delta *= hidden2_y * (1 - hidden2_y)
            self.weights[1] = self.weights[1] - (delta * np.dot(self.weights[2], hidden1_y.reshape(1, self.neuron_unit)).T)
            self.biases[1] = self.biases[1] - (delta * self.weights[2].reshape(self.neuron_unit))

            delta *= hidden1_y * (1 - hidden1_y)
            weight = np.dot(self.weights[1], self.weights[2])
            self.weights[0] = self.weights[0] - (delta * np.dot(weight, self.train_datas[i].reshape(1,self.input_size)).T)
            self.biases[0] = self.biases[0] - (delta * weight.reshape(self.neuron_unit))

        return np.mean(np.array(loss))

    def tanh_neuron_network(self):
        loss = []
        for i in range(len(self.train_datas)):
            hidden1_y = np.tanh(np.dot(self.weights[0].T, self.train_datas[i]) + self.biases[0])
            #hidden1_y = self.drop_out(hidden1_y)
            hidden2_y = np.tanh(np.dot(self.weights[1].T, hidden1_y) + self.biases[1])
            #hidden2_y = self.drop_out(hidden2_y)
            output_y = np.dot(self.weights[2].T, hidden2_y) + self.biases[2]
            error = output_y[0] - self.train_labels[i]
            loss.append(error ** 2 / 2)

            delta = self.learning_rate * error
            self.weights[2] = self.weights[2] - (delta * hidden2_y).reshape(self.neuron_unit, 1)
            self.biases[2] = self.biases[2] - delta

            delta *= (1 - hidden2_y * hidden2_y)
            self.weights[1] = self.weights[1] - (delta * np.dot(self.weights[2], hidden1_y.reshape(1, self.neuron_unit)).T)
            self.biases[1] = self.biases[1] - (delta * self.weights[2].reshape(self.neuron_unit))

            delta *= (1 - hidden1_y * hidden1_y)
            weight = np.dot(self.weights[1], self.weights[2])
            self.weights[0] = self.weights[0] - (delta * np.dot(weight, self.train_datas[i].reshape(1,self.input_size)).T)
            self.biases[0] = self.biases[0] - (delta * weight.reshape(self.neuron_unit))

        return np.mean(np.array(loss))

    def relu_neuron_network(self):
        loss = []
        for i in range(len(self.train_datas)):
            hidden1_y = np.dot(self.weights[0].T, self.train_datas[i]) + self.biases[0]
            hidden1_y[hidden1_y < 0] = 0
            hidden2_y = np.dot(self.weights[1].T, hidden1_y) + self.biases[1]
            hidden2_y[hidden2_y < 0] = 0
            output_y = np.dot(self.weights[2].T, hidden2_y) + self.biases[2]
            error = output_y[0] - self.train_labels[i]
            loss.append(error ** 2 / 2)

            delta = self.learning_rate * error
            self.weights[2] = self.weights[2] - (delta * hidden2_y).reshape(self.neuron_unit, 1)
            self.biases[2] = self.biases[2] - delta

            delta *= np.int64(hidden2_y>0)
            self.weights[1] = self.weights[1] - (delta * np.dot(self.weights[2], hidden1_y.reshape(1, self.neuron_unit)).T)
            self.biases[1] = self.biases[1] - (delta * self.weights[2].reshape(self.neuron_unit))

            delta *= np.int64(hidden1_y>0)
            weight = np.dot(self.weights[1], self.weights[2])
            self.weights[0] = self.weights[0] - (delta * np.dot(weight, self.train_datas[i].reshape(1,self.input_size)).T)
            self.biases[0] = self.biases[0] - (delta * weight.reshape(self.neuron_unit))

        return np.mean(np.array(loss))

    def linear_neuron_network(self):
        loss = []
        for i in range(len(self.train_datas)):
            hidden1_y = np.dot(self.weights[0].T, self.train_datas[i]) + self.biases[0]
            hidden2_y = np.dot(self.weights[1].T, hidden1_y) + self.biases[1]
            output_y = np.dot(self.weights[2].T, hidden2_y) + self.biases[2]
            error = output_y[0] - self.train_labels[i]
            loss.append(error ** 2 / 2)

            delta = self.learning_rate * error
            self.weights[2] = self.weights[2] - (delta * hidden2_y).reshape(self.neuron_unit, 1)
            self.biases[2] = self.biases[2] - delta

            self.weights[1] = self.weights[1] - (delta * np.dot(self.weights[2], hidden1_y.reshape(1, self.neuron_unit)).T)
            self.biases[1] = self.biases[1] - (delta * self.weights[2].reshape(self.neuron_unit))

            weight = np.dot(self.weights[1], self.weights[2])
            self.weights[0] = self.weights[0] - (delta * np.dot(weight, self.train_datas[i].reshape(1,self.input_size)).T)
            self.biases[0] = self.biases[0] - (delta * weight.reshape(self.neuron_unit))

        return np.mean(np.array(loss))

    def drop_out(self, x):
        work_prod = 1 - self.dropout_prob
        sample = np.random.binomial(n=1, p=work_prod, size=x.shape)
        x *= sample             # 将某些神经元的输出置零
        x *= 1/(work_prod)      # 放大剩下神经元的输出
        return x

    # def calculate_average_loss(self):
    #     loss = []
    #     for i in range(len(self.train_datas)):
    #         hidden1_y = np.dot(self.weights[0].T, self.train_datas[i]) + self.biases[0]
    #         hidden1_y[hidden1_y < 0] = 0
    #         hidden2_y = np.dot(self.weights[1].T, hidden1_y) + self.biases[1]
    #         hidden2_y[hidden2_y < 0] = 0
    #         output_y = np.dot(self.weights[2].T, hidden2_y) + self.biases[2]
    #         error = output_y[0] - self.train_labels[i]
    #         loss.append(error ** 2 / 2)
    #     return np.mean(np.array(loss))

    def calculate_result(self):
        for i in range(len(self.test_datas)):
            hidden1_y = np.tanh(np.dot(self.weights[0].T, self.test_datas[i]) + self.biases[0])
            hidden2_y = np.tanh(np.dot(self.weights[1].T, hidden1_y) + self.biases[1])
            output_y = np.dot(self.weights[2].T, hidden2_y) + self.biases[2]
            self.predict_labels.append(output_y[0])

    def save(self, file_path):
        np.savez(file_path, self.weights[0],self.weights[1],self.weights[2],self.biases[0],self.biases[1],self.biases[2])

    def restore(self, file_path):
        weights_biases = np.load(file_path)
        self.weights[0] = weights_biases["arr_0"]
        self.weights[1] = weights_biases["arr_1"]
        self.weights[2] = weights_biases["arr_2"]
        self.biases[0] = weights_biases["arr_3"]
        self.biases[1] = weights_biases["arr_4"]
        self.biases[2] = weights_biases["arr_5"]


if __name__ == "__main__":
    np.random.seed()
    nn = NN()
    nn.read("data/回归")
    nn.initial()
    nn.restore("check_point/回归/data_tag.npz")
    for i in range(500):
        print(str(i) + " epoch loss: " + str(nn.tanh_neuron_network()))
        if i % 50 == 0:
            nn.save("check_point/回归/data_tag.npz")
            print("save model")
    nn.calculate_result()
    nn.write("data/回归/16337250_0.txt")
