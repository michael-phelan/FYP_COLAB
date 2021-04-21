import tensorflow as tf

from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
import pickle
import sys
import numpy as np
from input import DataInput
from sklearn import metrics
from matplotlib import pyplot as plt
import random
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class GRU4Rec(object):

    def __init__(self, user_count, item_count, hidden_units, l2_penalty=1e-5):
        self.user_count = user_count
        self.item_count = item_count
        self.hidden_units = hidden_units = 64

        self.u = tf.placeholder(tf.int32, [None,]) # [B]
        self.i = tf.placeholder(tf.int32, [None,]) # [B]
        self.y = tf.placeholder(tf.float32, [None,]) # [B]
        self.hist_i = tf.placeholder(tf.int32, [None, None]) # [B, T]
        self.sl = tf.placeholder(tf.int32, [None,]) # [B]
        self.lr = tf.placeholder(tf.float64, [])


        # user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_units])
        self.item_emb_w = item_emb_w = tf.Variable(tf.random_normal([item_count, hidden_units], stddev=0.1), name='item_emb_w')
        self.item_b = item_b = tf.Variable(tf.random_normal([item_count], stddev=0.1), name='item_b')

        # ic = tf.gather(cate_list, self.i)
        i_emb = tf.concat([
            tf.nn.embedding_lookup(item_emb_w, self.i),
            # tf.nn.embedding_lookup(cate_emb_w, ic),
            ], 1)
        i_b = tf.gather(item_b, self.i)

        h_emb = tf.nn.embedding_lookup(item_emb_w, self.hist_i)

        # uni-directional rnn
        rnn_output, _ = tf.nn.dynamic_rnn(
            build_cell(hidden_units), h_emb, self.sl, dtype=tf.float32)

        hist = extract_axis_1(rnn_output, self.sl-1)
        hist = tf.layers.dense(hist, hidden_units)

        u_emb = hist

        with tf.variable_scope("estimation"):
            pre_values = tf.reduce_sum(tf.multiply(u_emb, i_emb), 1) + i_b
            self.pre_sig_value = tf.nn.sigmoid(pre_values)  # [B]

        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=pre_values))

        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step+1)


        trainable_params = tf.trainable_variables()
        self.opt = tf.train.AdamOptimizer(learning_rate=1e-3)
        # self.opt = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients = gradients
        # clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
        self.train_op = self.opt.apply_gradients(
            zip(clip_gradients, trainable_params), global_step=self.global_step)

    def train_step(self, sess, uij):
        loss, _, pre_values = sess.run([self.loss, self.train_op, self.pre_sig_value],
                feed_dict={ # (u, i, y, hist_i, sl)
                    self.u: uij[0],
                    self.i: uij[1],
                    self.y: uij[2],
                    self.hist_i: uij[3],
                    # self.hist_t: uij[4],
                    self.sl: uij[4],
                    })
        return loss, pre_values


    def test_step(self, sess, uij, cur_item_id=-1): # uid, hist_i, sl):
        return sess.run(self.pre_sig_value,
                feed_dict={
                    self.u: uij[0],
                    self.i: uij[1],
                    self.hist_i: uij[3],
                    self.sl: uij[4],
                    })


    def eval(self, sess, saver, val_set, model_path, is_test=0):
        # val_set = val_set[:128]
        pre_values, tru_values = [], []
        for iter, uij in DataInput(val_set, 128, self.item_count):
            # print(model.test(sess, uij))
            pre_values.extend(list(self.test_step(sess, uij)))
            tru_values.extend(uij[2])
        # print(tru_values[:100], pre_values[:100])
        # exit(1)
        mse = float(np.mean((np.array(pre_values) - np.array(tru_values)) ** 2))

        auc_scores = []
        for i in range(len(pre_values) // 2):
            auc_scores.append(metrics.roc_auc_score(tru_values[i * 2:i * 2 + 2], pre_values[i * 2:i * 2 + 2]))
        auc_score = float(np.mean(auc_scores))

        # if self.mse > mse and is_test == 1 :
        if auc_score > self.best_auc and is_test==1:
            # self.mse = mse
            self.best_auc = auc_score
            saver.save(sess, save_path=model_path)
            print(" save at: ", model_path, flush=True)

        return auc_score, mse, pre_values


    def train(self, sess, saver, train_set, val_set, model_path, epoch,item_count, for_poison=0):
        start_time = time.time()
        self.mse = 1e8
        self.best_auc = 0.0
        print("begin train.....", flush=True)

        train_batch_size = 64   # *20   # len(train_set) # ????  # int(len(y_tr) * (1 - self.validation_frac))
        # train_batch_size = len(train_set)

        auc, mse, _ = self.eval(sess, saver, val_set, model_path, is_test=1)
        print('val auc: %.4f mse: %.4f' % (auc, mse))

        # lr = 1.0;
        # print("lr: %.4f" % (lr))
        loss_sum = 0.0
        for epoch_num in range(epoch):
            random.shuffle(train_set)

            for _, uij in DataInput(train_set, train_batch_size, item_count):
                loss, pre_values_ = self.train_step(sess, uij)
                loss_sum += loss

                if self.global_step.eval(session=sess) % 100 == 0:
                    if for_poison==0:
                        val_auc, mse, _ = self.eval(sess, saver, val_set, model_path, is_test=1)
                    else:
                        val_auc, mse, _ = 0,0,0


                    dis = 0.0
                    print('Epoch %d Global_step %d\tTrain_loss: %.4f\tdis:%.4f\tval_AUC: %.4f\tbest_val_AUC: %.4f\tMSE: %.4f\ttime: %.4f' %
                        ( epoch_num, self.global_step.eval(session=sess),
                         loss_sum / 100, dis, val_auc, self.best_auc, mse, time.time()-start_time)) #, flush=True)
                    loss_sum = 0.0
                    if val_auc==1.0:
                        return 0
                self.global_epoch_step_op.eval(session=sess)



def extract_axis_1(data, ind):
    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)
    return res


def build_single_cell(hidden_units):
    cell_type = GRUCell
    # cell_type = GRUCell
    cell = cell_type(hidden_units)
    return cell

def build_cell(hidden_units, depth=1):
    cell_list = [build_single_cell(hidden_units) for i in range(depth)]
    return MultiRNNCell(cell_list)



if __name__=="__main__":
    if len(sys.argv) != 2:
        print("args Error!")
        exit(1)
    data_set = int(sys.argv[1])

    dataname = "Steam"
    attack_user_num = 50
    env_path = "../%d_7_%d" % (data_set, attack_user_num) + "_env" + "/ckpt"

    with open("../data/dataset_" + dataname + ".pkl", "rb") as f:
        train_set = pickle.load(f)
        val_set = pickle.load(f)
        test_set = pickle.load(f)
        _ = pickle.load(f)
        user_count, item_count, _ = pickle.load(f)
    user_count += attack_user_num
    item_count += 8  # 8个target item

    ratings = [()]
    for i in range(5):
        print(train_set[i], flush=True)

    gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=0.5)

    graph_env = tf.Graph()
    with graph_env.as_default():
        env = GRU4Rec(user_count, item_count, 128)
        sess_env = tf.Session(graph=graph_env, config=tf.ConfigProto(gpu_options=gpu_options))
        sess_env.run(tf.global_variables_initializer())
        sess_env.run(tf.local_variables_initializer())
        saver_env = tf.train.Saver()
    env.train(sess_env, saver_env, train_set, val_set, env_path, epoch=3, item_count= item_count)