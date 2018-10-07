import IO_regression
import numpy as np
import datetime

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
        self.weights = [np.random.randn(6, 6) / np.sqrt(6), np.random.randn(6, 1) / np.sqrt(6)]
        self.biases = [np.zeros(6), 0]

    def sigmoid(self, x):
        return 1 / (1+np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1-x)

    def neuron_network(self):
        loss = []
        for i in range(len(self.train_datas)):
            hidden_y = self.sigmoid(np.dot(self.weights[0].T, self.train_datas[i]) + self.biases[0])
            output_y = np.dot(self.weights[1].T, hidden_y) + self.biases[1]
            error = output_y[0] - self.train_labels[i]
            loss.append(error ** 2 / 2)
            self.weights[1] = self.weights[1] - (self.learning_rate * error * hidden_y).reshape(6,1)
            self.biases[1] = self.biases[1] - (self.learning_rate * error)
            self.weights[0] = self.weights[0] - (self.learning_rate * error * hidden_y * (1-hidden_y) * np.dot(self.weights[1], self.train_datas[i].reshape(1, 6)).T)
            self.biases[0] = self.biases[0] - (self.learning_rate * error * hidden_y * (1 - hidden_y) * self.weights[1].reshape(6))
        return np.mean(np.array(loss))

    def calculate_result(self):
        for i in range(len(self.test_datas)):
            hidden_y = self.sigmoid(np.dot(self.weights[0].T, self.train_datas[i]) + self.biases[0])
            output_y = np.dot(self.weights[1].T, hidden_y) + self.biases[1]
            self.predict_labels.append(output_y[0])

if __name__ == "__main__":
    np.random.seed()
    nn = NN()
    nn.read("data/回归")
    nn.initial()
    nn.neuron_network()
    for i in range(200):
        print(str(i) + " epoch: " + str(nn.neuron_network()))
    nn.calculate_result()
    nn.write("data/回归/16337250_3.txt")
