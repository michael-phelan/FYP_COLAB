#coding=utf-8
import tensorflow as tf
# import threading
import random
import operator
from input import DataInput
from sklearn import metrics
import time
import pickle
import numpy as np
import os
import csv
from rank.GRU4Rec import GRU4Rec


class Environment():
    # Ranker for recommender system
    def __init__(self, sample_order, para, poison, rewards, test_args, epoch=-1):
        self.sample_order = sample_order
        self.para = para
        self.poison = poison
        self.rewards = rewards
        self.test_args = test_args

        self.rec_name = para["rec_name"]
        self.rec_type = para["rec_type"]
        self.base_dir = self.test_args[:3] + "_" + str(50) + "_env"
        self.env_path = self.base_dir + "/ckpt"

        os.makedirs(self.base_dir, exist_ok=True)

        self.user_count, self.item_count, self.cate_count, self.cate_list = poison.get_user_count(), poison.get_item_count(), poison.get_cate_count(), poison.get_cate_list()

        gpu_options = tf.GPUOptions(allow_growth=True) # , per_process_gpu_memory_fraction=0.5)
        # Env Graph Definition
        self.graph_env = tf.Graph() # "Env_id_"+self.threadId
        with self.graph_env.as_default():
            self.env = GRU4Rec(self.user_count-self.para["attack_user_num"]+50, self.item_count, para["env_hidden_size"])

            self.sess_env = tf.Session(graph=self.graph_env, config=tf.ConfigProto(gpu_options=gpu_options))
            self.sess_env.run(tf.global_variables_initializer())
            self.sess_env.run(tf.local_variables_initializer())
            self.saver_env = tf.train.Saver()


    def restore(self, saver, sess, path):
        self.saver_env.restore(self.sess_env, path)  #

    def save(self, saver, sess, path):
        self.saver_env.save(self.sess_env, path)

    def run(self):
        # tf.debugging.set_log_device_placement(True)
        # with tf.device('/CPU:0'):
        if self.para["attack_type"]==-1:
            exit(1)

        else:
            print("Env is restore from: ", self.env_path, flush=True)
            path = self.env_path
            # self.saver_env.restore(self.sess_env, path)
            #self.save(self.saver_env, self.sess_env, path)
            self.restore(self.saver_env, self.sess_env, path)
            
            #print("Attack type: ", self.para["attack_type"])

            self.para["env_epoch"] = 1
            self.para["env_train_batch_size"] = 512


            if self.para["data_set"] == "Steam":
                self.para["env_epoch"] = 15
                self.para["env_train_batch_size"] = 512

            print("fine-tune env_epoch: ", self.para["env_epoch"], flush=True)

            self.poison.train_set = self.poison.attack_set
            print("number of the train poisoning data: %d" % (len(self.poison.train_set)), flush=True)

        self.best_auc_score = -1e8


        self.train_env(self.sess_env, self.env, self.saver_env)
        print("train_env finished!", flush=True)


        test_score, hit_num = self.eval(self.sess_env, self.env, self.saver_env, sample_order=int(self.sample_order), is_test=1)
        print('sample_order %d after poison, test score: %.4f hit num: %d' %(self.sample_order, test_score, hit_num), flush=True)

        # with open('hit_num.csv', 'a', newline='') as csvfile:
        #     fieldnames = ["hit_num"]
        #     filewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #     filewriter.writerow({"hit_num": hit_num})

        self.rewards[int(self.sample_order)] = hit_num

    def train_env(self, sess_env, env, saver):
        start_time = self.start_time
        train_set = self.poison.get_train_set();

        train_batch_size = self.para["env_train_batch_size"];
        loss_sum = 0.0

        for epoch in range(self.para["env_epoch"]):
            input = DataInput(train_set, train_batch_size, self.item_count)

            for _, uij in input:
                loss, pre_values_ = env.train_step(sess_env, uij)
                loss_sum += loss

                env.global_epoch_step_op.eval(session=sess_env)


    def eval(self, sess, model, saver, sample_order=-1, is_test=0):
        start_time = self.start_time
        def test_auc(data):
            pre_values, tru_values = [], []
            for iter, uij in DataInput(data, self.para["env_test_batch_size"], self.item_count):
                pre_values.extend(list(model.test_step(sess, uij)))
                tru_values.extend(uij[2])

            auc_scores = []
            for i in range(len(pre_values) // 2):
                auc_scores.append(metrics.roc_auc_score(tru_values[i * 2:i * 2 + 2], pre_values[i * 2:i * 2 + 2]))
            val_auc_score = float(np.mean(auc_scores))

            return val_auc_score

        if is_test==0:
            val_auc_score = test_auc(data=self.poison.get_val_set())
        else:
            val_auc_score = 0.0 #

        if (self.best_auc_score < val_auc_score) and is_test==0 and self.para["attack_type"]==2:
            self.best_auc_score = val_auc_score
            path = self.env_path
            # saver.save(sess, save_path= path)
            self.save(saver, sess, path)
            print("Save at: ", path)

        if is_test==1:
            print("begin is_test. time: %.4f"%(time.time()-start_time), flush=True)
            # test Hit     target_items
            all_items = [j for j in range(self.item_count - 8)]
            def test_hit_num(data, batch_size = 1):
                input_time1 = input_time2 = input_time3 = 0.0
                hit_num = 0
                top_ks = []
                for iter, uij in DataInput(data, batch_size, self.item_count):
                    # print("len(uij[0]): ", len(uij[0]))
                    if iter==1 or iter % (len(data)//batch_size//6) == 0: print("test id: %d/%d, hit num: %d, time: %.4f" % (iter*batch_size, len(data), hit_num, time.time()-start_time), flush=True)

                    if self.para["data_set"] == "Steam":
                        if random.random() > 1 : continue

                    uij_ = [[], [], [], [], []]  # (u, i)
                    # candidate_sets = []
                    item_ids = []
                    for i in range(len(uij[0])):

                        user_id = uij[0][i]
                        item_id = uij[1][i]

                        hist = uij[3][i]
                        hist_sl = uij[4][i] #
                        y = uij[2][i]
                        if y==0: continue

                        candidate_set = [j for j in range(self.item_count-8, self.item_count)]
                        candidate_set.extend(random.sample(all_items, 100-8))

                        # Rank score
                        for candi in candidate_set:
                            uij_[0].append(user_id)
                            uij_[1].append(candi)
                            # uij[2].append(0)
                            uij_[3].append(hist)
                            uij_[4].append(hist_sl)
                            item_ids.append(item_id)

                    pre_values = list(model.test_step(sess, uij_))

                    # for i in range(len(uij[0])//2):
                    for i in range(len(pre_values)//100):
                        single_pre_values = pre_values[i*100:(i+1)*100]
                        # single_candidate_set = candidate_sets[i: (i+1)*100]
                        single_candidate_set = uij_[1][i*100: (i+1)*100]

                        top_list_values = []
                        for j in range(len(single_candidate_set)):
                            if single_pre_values[j]>0:
                                top_list_values.append((single_candidate_set[j], single_pre_values[j]))

                        top_k = 10
                        top_ks.append(top_k)

                        top_list_values = sorted(top_list_values, key=operator.itemgetter(1,0))[-top_k:]

                        top_list = set([t[0] for t in top_list_values])
                        # get top_k
                        hit_num += len(set(self.poison.get_target_items()) & top_list)
                    # print("Hit num in the val set: %d" % (hit_num))  # , flush=True)
                print("input_time: %.4f %.4f %.4f"%(input_time1,input_time2,input_time3), flush=True)
                print("top_ks[:10]: ", top_ks[:10], flush=True)

                return hit_num
            hit_num = hit_num1 = test_hit_num(self.poison.get_val_set(), batch_size=32)

        else:
            hit_num=0

        return val_auc_score, hit_num