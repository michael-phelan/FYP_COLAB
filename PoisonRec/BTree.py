#coding=utf-8
import tensorflow as tf
import random
import time
import pickle
import operator
from input import DataInput
from sklearn import metrics
import numpy as np
import numpy as np
from matplotlib import pyplot as plt


random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

class BTree(object): # build item tree


    def __init__(self, user_count, item_count, item_nums):
        self.user_count = user_count
        self.item_count = item_count
        self.item_nums = item_nums

        item_id_nums = []
        for i in range(item_count):
            item_id_nums.append((i, item_nums[i]))
        item_id_nums = sorted(item_id_nums, key=operator.itemgetter(1))

        self.original_btree = [i for i in range(item_count)]
        self.target_btree = [i+item_count for i in range(8)]   #
        for i in range(item_count):
            self.original_btree[i] = item_id_nums[i][0]

        print("user_count: %d, item_count: %d, get original/target_tree finished!" %(user_count, item_count), flush=True)


    def view_tree(self):
        _ = 0


if __name__=="__main__":
    _ = 0
