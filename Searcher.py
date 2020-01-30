import numpy as np
from eunjeon import Mecab
import math


class SearchEngine:
    def __init__(self):
        file = open('wiki_output', 'r', encoding='utf-8')
        self.Docs = file.read().split('\a')

        self.Tagger = Mecab()

        words = open('titles', 'r', encoding='utf-8').read().split('\t')
        self.Titles = np.array(words, dtype='<U20')
        print(len(self.Titles), len(self.Docs))
        #input()

        self.Titles_ = np.array(words, dtype='<U20')
        self.args = self.Titles_.argsort()
        self.Titles_.sort()

        dic_file = open('entity_dic', 'r', encoding='utf-8')
        words = dic_file.read().split('\n')
        self.dic_entity = np.array(words)
        self.dic_entity.sort()

        self.indexer_pointer = np.load('indexer_pointer.npy')
        self.indexer_frequency = np.load('indexer_frequency.npy')
        self.indexer_idf = np.load('idf.npy')
        self.doc_length = np.load('doc_length.npy')

        print(self.indexer_pointer.shape, self.indexer_idf.shape)

    def search_doc(self, title):
        idx = self.Titles_.searchsorted(title)

        if idx == self.Titles_.shape[0]:
            return -1

        if self.Titles_[idx] == title:
            return self.args[idx]

        return -1

    def search(self, word):
        idx = self.dic_entity.searchsorted(word)

        if idx == self.dic_entity.shape[0]:
            return -1
        if word == self.dic_entity[idx]:
            return idx
        return -1

    def search_document(self, doc, top_Document_Number=5):
        score = np.zeros(shape=[self.Titles.shape[0]], dtype=np.float32)
        TK = self.Tagger.morphs(doc)
        pos = self.Tagger.pos(doc)

        for i in range(len(TK)):
            if pos[i][1][0] == 'N' or pos[i][1][0] == 'U':
                idx = self.search(TK[i])

                if idx != -1:
                    for j in range(50):
                        if self.indexer_pointer[idx, j] != 0:
                            try:
                                score_ = self.indexer_idf[idx] * self.indexer_frequency[idx, j] * 2.2
                                divider = self.indexer_frequency[idx, j] + 1.2 * \
                                          (1 - 0.7 + 0.7 * self.doc_length[self.indexer_pointer[idx, j]])

                                score[self.indexer_pointer[idx, j]] += score_ / divider
                            except:
                                0
        arg = score.argsort()
        arg = list(reversed(arg))

        return arg[0:top_Document_Number], score
