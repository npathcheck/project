from gensim.models import Word2Vec
import numpy as np

train_tags = []
test_tags = []
read_object = open("data/回归/trainTag.txt", 'r', encoding='UTF-8')
for line in read_object.readlines():
    train_tags.append(line.strip().split())
read_object.close()
read_object = open("data/回归/testTag.txt", 'r', encoding='UTF-8')
for line in read_object.readlines():
    test_tags.append(line.strip().split())
read_object.close()


model = Word2Vec(train_tags + test_tags, size=14, min_count=1)

train_tags_array = []
for train_tag in train_tags:
    train_tag_array = np.zeros(14)
    for tag in train_tag:
        train_tag_array += model[tag]
    train_tag_array /= len(train_tag)
    train_tags_array.append(train_tag_array)
train_tags_array = np.array(train_tags_array)

test_tags_array = []
for test_tag in test_tags:
    test_tag_array = np.zeros(14)
    for tag in test_tag:
        test_tag_array += model[tag]
        test_tag_array /= len(test_tag)
    test_tags_array.append(test_tag_array)
test_tags_array = np.array(test_tags_array)

np.savez("check_point/回归/nn/tags.npz", train_tags_array, test_tags_array)
