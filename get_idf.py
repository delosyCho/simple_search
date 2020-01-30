import numpy as np
import math

file = open('wiki_morph', 'r', encoding='utf-8')
dic_file = open('entity_dic', 'r', encoding='utf-8')
words = dic_file.read().split('\n')
dic_entity = np.array(words)
dic_entity.sort()

def search(word):
    idx = dic_entity.searchsorted(word)

    if idx == dic_entity.shape[0]:
        return -1
    if word == dic_entity[idx]:
        return idx
    return -1


idf = np.zeros(shape=[dic_entity.shape[0]], dtype=np.float32)
idf_ = np.zeros(shape=[dic_entity.shape[0]], dtype=np.int32)

docs = file.read().split('\a')
print(len(docs))
input()

count = np.zeros(shape=[dic_entity.shape[0]], dtype=np.int32)

for i in range(len(docs) - 1):
    frequency = np.zeros(shape=[dic_entity.shape[0]], dtype=np.int32)
    checker = np.zeros(shape=[dic_entity.shape[0]], dtype=np.int32)
    vocab_list = []

    if i % 50 == 0:
        print(i, '/', len(docs))

    doc = docs[i].split('\t')[1]

    TK = doc.split(' ')

    for k in range(len(TK)):
        idx = search(TK[k])

        if idx != -1:
            frequency[idx] += 1
            if checker[idx] == 0:
                checker[idx] = 1
                vocab_list.append(idx)

    for k in vocab_list:
        if frequency[k] > 0:
            idf_[k] += 1

for i in range(dic_entity.shape[0]):
    if idf_[i] > 0:
        idf[i] = (len(docs) - 1) / idf_[i]
        idf[i] = math.log(idf[i])

np.save('idf', idf)

file.close()
