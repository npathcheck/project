import numpy as np
import matplotlib.pyplot as plt

train_datas = []
test_datas = []
read_object = open("data/2/unclearTrainData.txt", 'r', encoding='UTF-8')
for line in read_object.readlines():
    train_datas.append(line.strip().split())
read_object.close()
read_object = open("data/2/unclearTestData.txt", 'r', encoding='UTF-8')
for line in read_object.readlines():
    test_datas.append(line.strip().split())
read_object.close()

# data/2/clear      200
# data/2/unclear    500

lengths = [len(line) for line in train_datas+test_datas]
plt.hist(lengths, bins=25)
plt.show()