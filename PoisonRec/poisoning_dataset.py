#coding=utf-8
import random
import pickle
import numpy as np
import operator
import math
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes
import ctypes
from ctypes import c_wchar_p


random.seed(1234)

class Poison(object):
    def __init__(self, data_name, btree, attack_user_num, step, test_args):
        # self.attack_name = ["none", "random", "popular", "middle", "rl_based"]
        self.data_name = data_name
        self.btree = btree
        self.attack_user_num=attack_user_num
        self.attack_example_num = step
        self.test_args = test_args
        self.target_item_num = 8

        self.load_data()
        self.split_set()

    def load_data(self):
        # load data
        with open('./data/dataset_' + self.data_name + '.pkl', 'rb') as f:
            self.train_set = pickle.load(f)
            self.val_set = pickle.load(f)
            self.test_set = pickle.load(f)
            self.cate_list = pickle.load(f)
            self.user_count, self.item_count, self.cate_count = pickle.load(f)
    
            self.user_count += self.attack_user_num
            self.item_count += 8

        self.attack_set = []
        self.poisoning_data_saved = []

    def reset_attack(self):
        self.load_data()
        self.attack_set = []
        self.poisoning_data_saved = []



    def split_set(self):
        # popular items/item_count/cate_count
        item_nums = [0 for i in range(self.item_count)]
        self.item_nums = item_nums
        for t in self.train_set:
            if t[4] == 1:
                item_nums[t[3]] += 1
        item_nums_ = sorted(item_nums, reverse=True)

        self.target_items = [i for i in range(self.item_count-8, self.item_count)]
        self.target_items = self.target_items # [:2]   #

        split_popular = item_nums_[int(len(np.nonzero(item_nums_)[0]) * 0.1)]

        # print(split_target, split_popular)
        self.popular_items = []  # top 10%
        self.unpopular_items = []
        for i in range(self.item_count):
            if item_nums[i] >= split_popular:
                self.popular_items.append(i)
            elif item_nums[i] >= 0:
                self.unpopular_items.append(i)
        for t in self.target_items:
            self.unpopular_items.remove(t)
        print("len(popular target unpopular):", len(self.popular_items), len(self.target_items), len(self.unpopular_items)) #, flush=True)


        split_items = [[] for i in range(10)]
        split_num = [
            item_nums_[int(len(np.nonzero(item_nums_)[0]) * j/10)] for j in range(1, 10)
        ]
        split_num.append(0)
        filter_set = {};
        for t in self.target_items: filter_set[t]=1
        for i in range(self.item_count):
            if i not in filter_set:
                j=0
                while(1):
                    if item_nums[i] >= split_num[j]:
                        break
                    j+=1
                split_items[j].append(i)
        for i in range(1, 10):  # [1-9]
            if len(split_items[i])==0:
                split_items[i] = split_items[i-1]
        print("split item set length: ", [len(split_items[i]) for i in range(10)]) #, flush=True)
        self.split_items = split_items



    def get_train_set(self):
        return self.train_set
    def get_val_set(self):
        return self.val_set
    def get_test_set(self):
        return self.test_set
    def get_user_count(self):
        return self.user_count
    def get_item_count(self):
        return self.item_count
    def get_cate_list(self):
        return self.cate_list
    def get_cate_count(self):
        return self.cate_count
    def get_popular_items(self):
        return self.popular_items
    def get_unpopular_items(self):
        return self.unpopular_items
    def get_target_items(self):
        return self.target_items

    def copy_from(self, data):
        # self.train_set = list(data.get_train_set())
        # self.user_count = data.get_user_count()
        self.popular_items = list(data.get_popular_items())
        self.target_items = list(data.get_target_items())
        self.unpopular_items = list(data.get_unpopular_items())
        self.split_items = list(data.split_items)

    def set_attack_type(self, attack_type):
        self.attack_type = attack_type

    def get_poisoning_data(self, user_th):
        # popular_items dict, used for finding
        popular_items_dict = {}
        for t in self.popular_items:
            popular_items_dict[t] = 1

        # data poisoning
        poisoning_data = []
        poisoning_actions = []
        if self.attack_type == 0:    # none
            print("none attack")

        elif self.attack_type == 1: # random
            print("random attack")
            unpopular_items = list(np.copy(self.unpopular_items))
            unpopular_items.extend(self.popular_items)
            for i in range(int(self.attack_example_num//2)):
                from_item = random.sample(self.target_items, 1)[0]
                # to_item = random.sample(self.target_items, 1)[0]
                to_item = random.sample(unpopular_items, 1)[0]
                poisoning_data.extend([from_item, to_item])
                poisoning_actions.append(1)
                if to_item in popular_items_dict:
                    poisoning_actions.append(0)
                else:
                    poisoning_actions.append(2)


        elif self.attack_type == 2: # popular

            print("popular attack")
            for i in range(int(self.attack_example_num // 2)):
                from_item = random.sample(self.target_items, 1)[0]
                to_item = random.sample(self.popular_items, 1)[0]
                poisoning_data.extend([from_item, to_item])
                poisoning_actions.append(1)
                poisoning_actions.append(0)

        return poisoning_data, poisoning_actions

    def trans_samples(self, poisoning_data, attack_each_user_num, j):
        attack_set = []
        if self.attack_type > 0:
            pos_list = poisoning_data
            for t in range(1, len(pos_list)):
                hist = pos_list[:t]
                attack_set.append((self.user_count-self.attack_user_num+j, hist, [], pos_list[t], 1, -1))


        self.attack_set.extend(attack_set)


    def get_samples_for_A2DPA(self, pre_lists, b_id, attack_user_num, action_num=3):
        # pre_lists [B*20, 5, 12]
        self.batch_flags = [] #

        attack_set = []
        print("===============user_count: ", self.user_count, flush=True)
        for i in range(b_id*attack_user_num, (b_id+1)*attack_user_num):
            u_id = self.user_count - attack_user_num + (i - b_id*attack_user_num)
            item_list = []
            pre_list = pre_lists[i] # [T, 12]

            batch_flag = [0 for i in range(len(pre_list))] # [0 for i in range(attack_user_num)]
            for j in range(len(pre_list)):

                if pre_list[j][0]==0:  # target item
                    decoder_first_level_V = 0                   # 0 layer
                    # decoder_other_level_V = 2 - pre_list[j][1]  # 1 layer
                    decoder_other_level_V = 1 + pre_list[j][1]  # 1 layer
                    for h in range(2, 4):   # 15 nodes in the first 4 layers are all target items
                        base_V = math.pow(2, h) - 1
                        action = pre_list[j][h]
                        decoder_other_level_V = base_V + 2*(decoder_other_level_V - (math.pow(2, h - 1) - 1)) + action
                    location = decoder_other_level_V - 7
                    location = int(location)
                    item_id = self.btree.target_btree[location]

                else:   # original item
                    decoder_first_level_V = 15  # 0层
                    # decoder_other_level_V = 2 - pre_list[j][1]  # 1层
                    decoder_other_level_V = 1 + pre_list[j][1]  # 1层
                    for h in range(2, len(pre_list[j])):  #
                        base_V = math.pow(2, h) - 1
                        action = pre_list[j][h]
                        # print("decoder_other_level_V", decoder_other_level_V, base_V, action, decoder_other_level_V - math.pow(2, h - 1) + 1, flush=True)
                        decoder_other_level_V = base_V + 2 * (decoder_other_level_V - (math.pow(2, h - 1) - 1)) + action

                        if (decoder_other_level_V - base_V) < (self.item_count-8):
                            location = decoder_other_level_V - base_V
                            location = int(location)
                        else:
                            batch_flag[j] = 1
                            break
                    # print("location: %d" % location, flush=True)
                    item_id = self.btree.original_btree[location]
                item_list.append(item_id)
            self.batch_flags.append(batch_flag)
            for j in range(1, len(item_list)):
                t = item_list[j]
                hist = item_list[:j]
                attack_set.append((u_id, hist, [], t, 1))

                # neg = item_list[0]
                # while neg in item_list:
                #     neg = random.randint(0, self.item_count - 1)
                # attack_set.append((u_id, hist, [], neg, 0))
        self.attack_set = attack_set


if __name__ == "__main__":
    print("get poisoning data")
