#coding=utf-8
import numpy as np
import scipy.sparse as spyspa
import random

class DataInput: # s
    def __init__(self, data, batch_size, item_count):

        self.batch_size = batch_size
        self.data = data
        self.sumary_u_pos(data)
        self.item_count = item_count
        self.epoch_size = len(self.data) // self.batch_size
        if self.epoch_size * self.batch_size < len(self.data):
            self.epoch_size += 1
        self.i = 0

    def sumary_u_pos(self, data):
        self.u_pos = {}
        for t in data:
            if t[4]>0:
                if t[0] not in self.u_pos:
                    self.u_pos[t[0]] = []
                self.u_pos[t[0]].append(t[3])

    def __iter__(self):
        return self

    def __next__(self):

        if self.i == self.epoch_size:
            raise StopIteration

        ts = self.data[self.i * self.batch_size : min((self.i+1) * self.batch_size, len(self.data))]
        self.i += 1


        u, i, y, sl = [], [], [], []

        for t in ts:    # (reviewerID, hist, hist_t, neg_list[i],  0)
            # Increase positive sample
            u.append(t[0])
            i.append(t[3])
            # y.append(t[4])
            y.append(1)

            sl.append(len(t[1]))
            if len(t[1]) == 0:
                print(self.i, t)
                print("Error! len(hist)==0!", flush=True)
                exit(1)
            # y.append(1)

            # Randomly increase negataive samples
            i_negs = []
            for j in range(1):
                i_neg = random.randint(0, self.item_count-1)
                while(i_neg in self.u_pos[t[0]] or i_neg in i_negs):
                    i_neg = random.randint(0, self.item_count - 1)
                i_negs.append(i_neg)
                u.append(t[0])
                i.append(i_neg)
                y.append(0)
                sl.append(len(t[1]))
        max_sl = max(sl)

        hist_i = np.zeros([len(ts)*(1+1), max_sl], np.int64)
        for k, t in enumerate(ts):
            for l in range(len(t[1])):
                hist_i[2 * k][l] = t[1][l]
                hist_i[2 * k + 1][l] = t[1][l]
                # hist_i[2 * k + 2][l] = t[1][l]
                # hist_i[2 * k + 3][l] = t[1][l]


        return self.i, (u, i, y, hist_i, sl)
