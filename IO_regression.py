import pandas as pd
import numpy as np
import re


def clear(file_path, type):
    df = pd.read_excel(file_path+"/" + type + ".xlsx")
    datas = df.values[:,:6]
    tags = df.values[:,6]
    for i in range(len(tags)):
        tags[i] = [''.join(tag.split()) for tag in list(filter(lambda s: s and s.strip(), re.split(r'[\',\[\]]', tags[i])))]
    write_object = open(file_path + "/" + type + "Data.txt", 'w', encoding='UTF-8')
    for data in datas:
        write_object.write(' '.join('%s' % s for s in data) + '\n')
    write_object.close()
    write_object = open(file_path + "/" + type + "Tag.txt", 'w', encoding='UTF-8')
    for tag in tags:
        write_object.write(' '.join('%s' % s for s in tag) + '\n')
    write_object.close()
    if type == "train":
        labels = df.values[:, -1]
        write_object = open(file_path + "/" + type + "Label.txt", 'w', encoding='UTF-8')
        for label in labels:
            write_object.write(str(label) + '\n')
        write_object.close()

def read(file_path, type):
    datas = []
    tags = []
    read_object = open(file_path+"/" + type + "Data.txt", 'r', encoding='UTF-8')
    for line in read_object.readlines():
        datas.append(line.strip().split(' '))
    read_object.close()
    read_object = open(file_path + "/" + type + "Tag.txt", 'r', encoding='UTF-8')
    for line in read_object.readlines():
        tags.append(set(line.strip().split(' ')))
    read_object.close()
    if type == "train":
        labels = []
        read_object = open(file_path + "/" + type + "Label.txt", 'r', encoding='UTF-8')
        for line in read_object.readlines():
            labels.append(line.strip().split(' '))
        read_object.close()
        return np.array(datas).astype(np.float64), tags, np.array(labels).astype(np.float64)
    else:
        return np.array(datas).astype(np.float64), tags

def write(file_path, predict_labels):
    write_object = open(file_path, 'w', encoding='UTF-8')
    for predict_label in predict_labels:
        write_object.write(str(predict_label) + '\n')
    write_object.close()

if __name__ == "__main__":
    #clear("data/回归", "train")
    clear("data/回归", "test")
