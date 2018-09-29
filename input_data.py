import re
import numpy as np

def read(file_path):
    '''
    :param file_name: 读入文件的路径
    :return: 以二维数组形式存储的单词
    以逗号，句号，冒号，括号和空格分隔字符串
    '''
    datas = []
    file_object = open(file_path, 'r', encoding='UTF-8')
    for line in file_object.readlines()[:6]:
        line = re.sub(r'<.*?>', '',line.strip())
        datas.append(list(filter(None, (word.lower() for word in re.split(r'[,.:() ]', line)))))
    file_object.close()
    return datas

def k_fold_cross_validation(datas, k=10):
    '''
    :param datas: 以二维列表形式存储的数据集
    :param k: K折交叉验证的k
    :return: 以三维列表形式存储的k个训练集和k个验证集
    '''
    length = len(datas)
    train_datas = []
    validation_datas = []
    for i in range(k):
        train_datas.append(datas[:int(length / k * i)] + datas[int(length / k * (i + 1)):])
        validation_datas.append(datas[int(length / k * i):int(length / k * (i + 1))])
    return train_datas, validation_datas


if __name__ == "__main__":
    datas = read("data/2/trainData.txt")
    train_datas, validation_datas = k_fold_cross_validation(datas)