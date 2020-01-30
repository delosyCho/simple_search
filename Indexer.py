import numpy as np


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


N = 50

#검색에 사용될 문서
file = open('wiki_morph', 'r', encoding='utf-8')
docs = file.read().split('\a')

#[전체 키워드의 개수, 최대 저장 개수]
#해당 문서에서의 빈도수를 나타내는 배열
indexer_frequency = np.zeros(shape=[dic_entity.shape[0], N], dtype=np.int32)
#문서의 번호를 저장하는 배열
indexer_pointer = np.zeros(shape=[dic_entity.shape[0], N], dtype=np.int32)

#해당 키워드가 총 몇개의 문서에서 등장했는지 저장하는 배열
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
            point = np.array(indexer_frequency[k], dtype=np.int32).argmin()
            indexer_frequency[k, point] = frequency[k]
            indexer_pointer[k, point] = i

    count += checker

np.save('indexer_pointer', indexer_pointer)
np.save('indexer_frequency', indexer_frequency)
np.save('indexer_count', count)


file.close()
