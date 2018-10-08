import re
import enchant
import math
import numpy as np
from collections import Counter
from nltk.stem import SnowballStemmer

def clear(file_path):
    '''
    数据清洗
    '''
    stemmer = SnowballStemmer("english")
    ecd = enchant.Dict("en_US")
    train_datas = []
    test_datas = []
    stop_words = []

    read_object = open("data/stoplist.csv", 'r', encoding='UTF-8')
    for line in read_object.readlines():
        stop_words.append(line.strip())
    read_object.close()

    step = 0
    read_object = open(file_path+"/trainData.txt", 'r', encoding='UTF-8')
    for line in read_object.readlines():
        line = re.sub(r'<.*?>', '',line.strip())
        line_words = list(filter(None, (stemmer.stem(word.lower()) for word in re.split(r'[-?/*&,;.:!()<>"`$#%@ ]', line))))
        for i in range(len(line_words)):
            if not ecd.check(line_words[i]):
                line_words[i] = (ecd.suggest(line_words[i])+[line_words[i]])[0].lower()
        train_datas.append([word for word in line_words if word not in stop_words])
        print("train: " + str(step))
        step += 1
    read_object.close()
    write_object = open(file_path+"/clearTrainData.txt", 'w', encoding='UTF-8')
    for train_data in train_datas:
        write_object.write(' '.join('%s' %s for s in train_data) + '\n')
    write_object.close()

    step = 0
    read_object = open(file_path + "/testData.txt", 'r', encoding='UTF-8')
    for line in read_object.readlines():
        line = re.sub(r'<.*?>', '', line.strip())
        line_words = list(filter(None, (stemmer.stem(word.lower()) for word in re.split(r'[-?/*&,;.:!()<>"`$#%@ ]', line))))
        for i in range(len(line_words)):
            if not ecd.check(line_words[i]):
                line_words[i] = (ecd.suggest(line_words[i])+[line_words[i]])[0].lower()
        test_datas.append([word for word in line_words if word not in stop_words])
        print("test: " + str(step))
        step += 1
    read_object.close()
    write_object = open(file_path + "/clearTestData.txt", 'w', encoding='UTF-8')
    for test_data in test_datas:
        write_object.write(' '.join('%s' %s for s in test_data) + '\n')
    write_object.close()


def resplit(file_path):
    train_datas = []
    test_datas = []

    read_object = open(file_path + "/trainData.txt", 'r', encoding='UTF-8')
    for line in read_object.readlines():
        line = re.sub(r'<.*?>', '', line.strip())
        train_datas.append(list(filter(None, (word.lower() for word in re.split(r'[-?/*&,;.:!()<>"`$#%@ ]', line)))))
    read_object.close()
    write_object = open(file_path + "/unclearTrainData.txt", 'w', encoding='UTF-8')
    for train_data in train_datas:
        write_object.write(' '.join('%s' % s for s in train_data) + '\n')
    write_object.close()

    read_object = open(file_path + "/testData.txt", 'r', encoding='UTF-8')
    for line in read_object.readlines():
        line = re.sub(r'<.*?>', '', line.strip())
        test_datas.append(list(filter(None, (word.lower() for word in re.split(r'[-?/*&,;.:!()<>"`$#%@ ]', line)))))
    read_object.close()
    write_object = open(file_path + "/unclearTestData.txt", 'w', encoding='UTF-8')
    for test_data in test_datas:
        write_object.write(' '.join('%s' % s for s in test_data) + '\n')
    write_object.close()



def read(file_path, type = "clear"):
    train_datas = []
    train_labels = []
    test_datas =[]
    read_object = open(file_path + "/" + type + "TrainData.txt", 'r', encoding='UTF-8')
    for line in read_object.readlines():
        train_datas.append(line.strip().split(' '))
    read_object.close()
    read_object = open(file_path + "/trainLabel.txt", 'r', encoding='UTF-8')
    for line in read_object.readlines():
        train_labels.append(int(line))
    read_object.close()
    read_object = open(file_path + "/" + type + "TestData.txt", 'r', encoding='UTF-8')
    for line in read_object.readlines():
        test_datas.append(line.strip().split(' '))
    read_object.close()
    return train_datas, train_labels, test_datas

def write(file_path, predict_labels):
    write_object = open(file_path, 'w', encoding='UTF-8')
    for predict_label in predict_labels:
        write_object.write(str(predict_label) + '\n')
    write_object.close()

def tf_idf(file_path):
    train_datas, train_labels, test_datas = read(file_path, "clear")
    key_train_datas = []
    key_test_datas = []
    datas = []
    for train_data in train_datas:
        datas += list(set(train_data))
    for test_data in test_datas:
        datas += list(set(test_data))
    datas_dict = Counter(datas)
    datas_len = len(train_datas) + len(test_datas)
    for train_data in train_datas:
        data_tfidf = []
        data_dict = Counter(train_data)
        for word in train_data:
            data_tfidf.append([-data_dict[word] * math.log(datas_len/datas_dict[word]), word])
        key_train_datas.append([word[1] for word in sorted(data_tfidf)[:50]])
    for test_data in test_datas:
        data_tfidf = []
        data_dict = Counter(test_data)
        for word in test_data:
            data_tfidf.append([-data_dict[word] * math.log(datas_len/datas_dict[word]), word])
        key_test_datas.append([word[1] for word in sorted(data_tfidf)[:50]])
    write_object = open(file_path + "/keyTrainData.txt", 'w', encoding='UTF-8')
    for key_train_data in key_train_datas:
        write_object.write(' '.join('%s' % s for s in key_train_data) + '\n')
    write_object.close()
    write_object = open(file_path + "/keyTestData.txt", 'w', encoding='UTF-8')
    for key_test_data in key_test_datas:
        write_object.write(' '.join('%s' % s for s in key_test_data) + '\n')
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
    resplit("data/5")