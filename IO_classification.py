import re
from nltk.stem import SnowballStemmer

def clear(file_path):
    '''
    数据清洗
    '''
    stemmer = SnowballStemmer("english")
    train_datas = []
    test_datas = []
    stop_words = []

    read_object = open("data/stoplist.csv", 'r', encoding='UTF-8')
    for line in read_object.readlines():
        stop_words.append(line.strip())
    read_object.close()

    i = 0
    read_object = open(file_path+"/trainData.txt", 'r', encoding='UTF-8')
    for line in read_object.readlines():
        line = re.sub(r'<.*?>', '',line.strip())
        line_words = list(filter(None, (word.lower() for word in re.split(r'[?/*&,;.:!-()<>" ]', line))))
        train_datas.append([stemmer.stem(word) for word in line_words if word not in stop_words])
        print("train: " + str(i))
        i += 1
    read_object.close()
    write_object = open(file_path+"/clearTrainData.txt", 'w', encoding='UTF-8')
    for train_data in train_datas:
        write_object.write(' '.join('%s' %s for s in train_data) + '\n')
    write_object.close()

    i = 0
    read_object = open(file_path + "/testData.txt", 'r', encoding='UTF-8')
    for line in read_object.readlines():
        line = re.sub(r'<.*?>', '', line.strip())
        line_words = list(filter(None, (word.lower() for word in re.split(r'[?/*&,;.:!-()<>" ]', line))))
        test_datas.append([stemmer.stem(word) for word in line_words if word not in stop_words])
        print("test: " + str(i))
        i += 1
    read_object.close()
    write_object = open(file_path + "/clearTestData.txt", 'w', encoding='UTF-8')
    for test_data in test_datas:
        write_object.write(' '.join('%s' %s for s in test_data) + '\n')
    write_object.close()


def read(file_path):
    train_datas = []
    train_labels = []
    test_datas =[]
    read_object = open(file_path + "/clearTrainData.txt", 'r', encoding='UTF-8')
    for line in read_object.readlines():
        train_datas.append(line.split(' '))
    read_object.close()
    read_object = open(file_path + "/trainLabel.txt", 'r', encoding='UTF-8')
    for line in read_object.readlines():
        train_labels.append(int(line))
    read_object.close()
    read_object = open(file_path + "/clearTestData.txt", 'r', encoding='UTF-8')
    for line in read_object.readlines():
        test_datas.append(line.split(' '))
    read_object.close()
    return train_datas, train_labels, test_datas

def write(file_path, predict_labels):
    write_object = open(file_path + "/16337250_4.txt", 'w', encoding='UTF-8')
    for predict_label in predict_labels:
        write_object.write(str(predict_label) + '\n')
    write_object.close()

def k_fold_cross_validation(train_datas, train_labels, k=10):
    '''
    :param train_datas：训练集数据, train_labels：训练集标签
    :param k: K折交叉验证的k
    :return 返回k个训练集和验证集
    '''
    length = len(train_datas)
    k_train_datas = []
    k_train_labels = []
    k_validation_datas = []
    k_validation_labels = []
    for i in range(k):
        k_train_datas.append(train_datas[:int(length / k * i)] + train_datas[int(length / k * (i + 1)):])
        k_train_labels.append(train_labels[:int(length / k * i)] + train_labels[int(length / k * (i + 1)):])
        k_validation_datas.append(train_datas[int(length / k * i):int(length / k * (i + 1))])
        k_validation_labels.append(train_labels[int(length / k * i):int(length / k * (i + 1))])
    return  k_train_datas, k_train_labels, k_validation_datas, k_validation_labels


if __name__ == "__main__":
    clear("data/2")