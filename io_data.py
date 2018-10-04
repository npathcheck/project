import re
import numpy as np

def read_train(train_data_path, train_label_path):
    '''
    :param train_data_path: 训练集数据的路径  train_label_path：训练集标签的路径
    :return: 以二维数组形式存储的单词
    以逗号，句号，冒号，括号和空格分隔字符串
    注意！没有去重。
    '''
    train_datas = []
    train_lables = []
    file_object = open(train_data_path, 'r', encoding='UTF-8')
    for line in file_object.readlines():
        line = re.sub(r'<.*?>', '',line.strip())
        train_datas.append(list(filter(None, (word.lower() for word in re.split(r'[,.:() ]', line)))))
    file_object.close()
    file_object = open(train_label_path, 'r', encoding='UTF-8')
    for line in file_object.readlines():
        train_lables.append(int(line))
    file_object.close()
    return train_datas, train_lables

def read_test(test_data_path):
    test_datas = []
    file_object = open(test_data_path, 'r', encoding='UTF-8')
    for line in file_object.readlines():
        line = re.sub(r'<.*?>', '',line.strip())
        test_datas.append(list(filter(None, (word.lower() for word in re.split(r'[,.:() ]', line)))))
    file_object.close()
    return test_datas

def write_test(test_label_path, predict_labels):
    file_object = open(test_label_path, 'w', encoding='UTF-8')
    for predict_label in predict_labels:
        file_object.write(str(predict_label) + '\n')
    file_object.close()

def k_fold_cross_validation(train_datas, train_lables, k=10):
    '''
    :param train_datas：训练集数据, train_lables：训练集标签
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
        k_train_labels.append(train_lables[:int(length / k * i)] + train_lables[int(length / k * (i + 1)):])
        k_validation_datas.append(train_datas[int(length / k * i):int(length / k * (i + 1))])
        k_validation_labels.append(train_lables[int(length / k * i):int(length / k * (i + 1))])
    return  k_train_datas, k_train_labels, k_validation_datas, k_validation_labels


if __name__ == "__main__":
    train_datas, train_lables = read_train("data/2/trainData.txt", "data/2/trainLabel.txt")
    k_train_datas, k_train_labels, k_validation_datas, k_validation_labels = k_fold_cross_validation(train_datas, train_lables)