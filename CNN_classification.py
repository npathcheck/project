from gensim.models.word2vec import Word2Vec
import numpy as np
import IO_classification

class CNN():

    train_datas = []
    train_labels = []
    test_datas = []
    predict_labels = []

    def read(self, file_path):
        self.train_datas, self.train_labels, self.test_datas = IO_classification.read(file_path)
        model = Word2Vec(self.train_datas)
        vectors = np.array([model[word] for word in (model.wv.vocab)])


if __name__ == '__main__':
    cnn = CNN()
    cnn.read("data/2")