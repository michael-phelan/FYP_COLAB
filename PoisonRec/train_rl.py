#coding=utf-8
import os
import gc
import time
import pickle
import random
import numpy as np
import tensorflow as tf
import sys
import copy
import operator
import math
from sklearn import metrics
from input import DataInput
from actor2 import A2DPA
from poisoning_dataset import Poison
# import threading
from environment import Environment
from BTree import BTree

random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)
start_time = time.time()
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# read args
if len(sys.argv) != 7:  # 0 7 20 20 1 0
    print("args Error!")
    exit(1)
data_set = int(sys.argv[1])
rec_type = int(sys.argv[2])
attack_user_num = int(sys.argv[3])
attack_step = int(sys.argv[4])
is_A2DPA = int(sys.argv[5])
if is_A2DPA==1:
    tree_depth = int(sys.argv[6])

para = {
    "act_train_batch_size": 16,
    "act_test_batch_size":  32,
    "act_hidden_size":      32,
    "act_epoch":            500,

    "env_train_batch_size": 8,
    "env_test_batch_size":  512,
    "env_hidden_size":      32,
    "env_epoch":            10,

    "attack_type":      7,
    "attack_name":      ["none", "random", "popular", ],

    "data_set":         ["Steam", "movielens", "movielens_modified_5_30", "movielens_modified_500_0", "movielens_modified_0_500", "movielens_modified_250_0", "movielens_modified_0_250", "movielens_modified_['Comedy']"],

    "rec_type":         0,  #
    "rec_name":         ["ItemPop", "CoVisitation","ItemKNN", "PMF", "BPR", "NeuMF",  "AutoRec", "GRU4Rec", "NGCF"],

    "attack_user_num":  20, # [5,10,|20|,30,40,50]
    "attack_step":      20, # [5,10,15,|20|,25,30]

    "is_pre_embedding": 0,
    "action_num":       1000,
    "tree_depth":       10,

    "top_k":            5,
}

para["data_set"] = para["data_set"][data_set]
para["rec_type"] = rec_type
para["attack_user_num"] = attack_user_num
para["attack_step"] = attack_step
para["is_A2DPA"] = is_A2DPA
if is_A2DPA==1:
    para["tree_depth"] = tree_depth
print(para, flush=True)


test_args = "%d_%d_%d_%d"%(data_set,rec_type,attack_user_num,attack_step)

print("test args: ", test_args, flush=True)


with open('./data/dataset_' + para["data_set"] + '.pkl', 'rb') as f:
    print("load dataset: ", para["data_set"], flush=True)
    train_set = pickle.load(f)
    val_set = pickle.load(f)
    test_set = pickle.load(f)
    cate_list = pickle.load(f)
    user_count, item_count, cate_count = pickle.load(f)
    print(user_count, item_count, flush=True)


max_tree_depth = math.log(item_count, 2)
if int(max_tree_depth) != max_tree_depth:
    max_tree_depth = int(max_tree_depth) + 1
max_tree_depth += 1
para["max_tree_depth"] = int(max_tree_depth)



if is_A2DPA==1:
    if tree_depth==0 or tree_depth>para["max_tree_depth"]:
        tree_depth = para["max_tree_depth"]
        para["tree_depth"] = int(tree_depth)



def load_popular(data_name):
    # load data
    with open('./data/dataset_' + data_name + '.pkl', 'rb') as f:
        train_set = pickle.load(f)
        val_set = pickle.load(f)
        test_set = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, item_count, cate_count = pickle.load(f)

    item_nums = [0 for i in range(item_count)]
    for t in train_set:
        if t[4] == 1:
            item_nums[t[3]] += 1
    # item_nums_ = sorted(item_nums, reverse=True)
    return user_count, item_count, item_nums

def train_act(graph_act, actor, sess_act, saver_act, btree):
    base_poison = Poison(data_name=para["data_set"], btree=btree, attack_user_num=para["attack_user_num"],
                    step=para["attack_step"], test_args=test_args)

    best_reward = 0.0
    for epoch in range(para["act_epoch"]):
        print("epoch: %d"%(epoch))

        # attack baselines
        baseline_num = len(para["attack_name"])
        base_rewards = [0 for i in range(baseline_num)]

        if epoch<=0:
            for i in range(0, baseline_num):

                if i>=3:    #
                    break

                poison = base_poison
                poison.reset_attack()
                poison.set_attack_type(i)        # set attack type
                for j in range(para["attack_user_num"]):
                    poisoning_data, base_list = poison.get_poisoning_data(j)
                    poison.trans_samples(poisoning_data, para["attack_step"], j)

                para["attack_type"] = i
                env = Environment(i, dict(para), poison, base_rewards, test_args, epoch=epoch)
                env.start_time = start_time
                env.run()
                gc.collect()

        print("baselines' rewards: ", base_rewards)
        # continue

        # RL-based attack
        para["attack_type"] = len(para["attack_name"])-1
        # pre_lists     # [B,5,20]
        pre_lists, pre_max_lists, scores, scores_mask, H, S = actor.get_predicted_output(sess_act)

        # base_probs for PPO method
        base_probs = []
        for j in range(len(scores_mask)):   # [32*10, 5, 12] / [32*10, 5, 12, 2]
            probs1 = []
            for k in range(len(scores_mask[j])):
                probs2 = []
                for l in range(len(scores_mask[j][k])):
                    a = pre_lists[j][k][l]
                    probs2.append(scores_mask[j][k][l][a])
                probs1.append(probs2)
            base_probs.append(probs1)  # [B*10,T,D]


        if para["tree_depth"] != para["max_tree_depth"]:
            pre_lists_add = []
            for i in range(len(pre_lists)):
                lists = []
                for j in range(len(pre_lists[i])):
                    pre = list(pre_lists[i][j])
                    pre.extend([1 if random.random()>=0.5 else 0 for _ in range(para["tree_depth"], para["max_tree_depth"])])
                    lists.append(pre)
                pre_lists_add.append(lists)
        else:
            pre_lists_add = pre_lists

        # train env and get the rewards (RecNum)
        rewards = [0 for i in range(para["act_train_batch_size"]+1)]
        batch_flags = []
        for i in range(para["act_train_batch_size"]):
            # re set values

            poison = base_poison
            poison.reset_attack()
            poison.set_attack_type(para["attack_type"])  # 7: rl
            poison.set_attack_type(7)  # 7: rl
            poison.get_samples_for_A2DPA(pre_lists_add, i, para["attack_user_num"])
            batch_flags.extend(poison.batch_flags)

            env = Environment(i+1, dict(para), poison, rewards, test_args, epoch=epoch)
            env.start_time = start_time
            env.run()
            gc.collect()
            print("actor %d finished. time: %.4f" % (i, time.time() - start_time), flush=True)

        rewards = rewards[1:]
        rewards = list(np.array(rewards)-base_rewards[0])
        # train actor
        user_rewards = []
        for t in rewards:
            user_rewards.extend([t for j in range(para["attack_user_num"])])
        user_rewards = np.array(user_rewards)
        # rewards = rewards - base_rewards[0] # improved RecNum
        if np.std(user_rewards)==0:
            print("Current policy Error! All Same! Zero? Please check it!")
            continue
        normd_rewards = (user_rewards - np.mean(user_rewards)) / np.std(user_rewards)
        print("improved rewards:", rewards)
        print("normd    rewards:", normd_rewards[0])

        #
        pre_lists_flag = []
        if tree_depth<=4:
            target_flag = [1 for i in range(tree_depth)]
        else:
            target_flag = [1,1,1,1]
            target_flag.extend([0 for i in range(4, len(pre_lists[0][0]))])
        original_flag = [1 for i in range(len(pre_lists[0][0]))]


        for i in range(len(pre_lists)): # [B*20,T,D]
            flags = []
            for j in range(len(pre_lists[i])):
                if pre_lists[i][j][0] == 0:
                    flags.append(target_flag)
                else:
                    if batch_flags[i][j]==1:
                        original_flag[-1] = 0
                        flags.append(original_flag)
                        original_flag[-1] = 1
                    else:
                        flags.append(original_flag)
            pre_lists_flag.append(flags)

        actor.train_step(sess_act, pre_lists, pre_lists_flag, base_probs, normd_rewards)

        max_rewards = [0, 0]

        if float(np.mean(rewards)) > best_reward:
            best_reward = float(np.mean(rewards)) # + none_rewards[0]

        print("Epoch %d, none: %d, random: %d, popular: %d, ADPA soft reward: %.1f, max: %d, time: %.1f" % (
                epoch, base_rewards[0], base_rewards[1], base_rewards[2], float(np.mean(rewards)), max_rewards[0],
                time.time() - start_time), flush=True)
        gc.collect()


if __name__ == '__main__':
    gpu_options = tf.GPUOptions(allow_growth=True) #  , per_process_gpu_memory_fraction=0.5)

    graph_act = tf.Graph()
    with graph_act.as_default():
        actor = A2DPA(attack_user_num=para["attack_user_num"], step=para["attack_step"], item_count=item_count,
                      tree_depth=para["tree_depth"], batch_size=para["act_train_batch_size"],
                      hidden_size=para["act_hidden_size"])
        sess_act = tf.Session(graph=graph_act, config=tf.ConfigProto(gpu_options=gpu_options))
        # writer = tf.summary.FileWriter("actor_graph/", sess_act.graph)
        sess_act.run(tf.global_variables_initializer())
        sess_act.run(tf.local_variables_initializer())
        saver_act = tf.train.Saver()


    user_count, item_count, item_nums = load_popular(para["data_set"])
    btree = BTree(user_count, item_count, item_nums)

    #tf.debugging.set_log_device_placement(True)
    
    train_act(graph_act, actor, sess_act, saver_act, btree)
    # test_act(graph_act, actor, sess_act, saver_act)
