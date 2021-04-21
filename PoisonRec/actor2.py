#coding=utf-8
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn_cell import LSTMStateTuple
import random
import math

random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)


def select_son_nodes(query, W_ref, H, hidden_size):
    output_l1 = tf.nn.relu(tf.matmul(query, W_ref)) # [B,H]

    output_l1 = tf.reshape(output_l1, shape=[-1, 1, hidden_size])   # [B,1,H]
    scores = tf.matmul(output_l1, tf.transpose(H, [0,2,1]))     # [B,1,H] * [B,H,2] = [B,1,2]
    scores = tf.reshape(scores, shape=[-1, 2])
    return scores, tf.nn.softmax(scores)    # [B,2]


class A2DPA(object):

    def __init__(self, attack_user_num, step, item_count, tree_depth, batch_size, hidden_size=32):
        self.batch_size = batch_size = batch_size * attack_user_num
        self.attack_user_number = attack_user_num
        self.hidden_size = hidden_size
        self.max_step = max_step = step  # could not be too large.
        self.item_count = item_count
        print("attack_user_num: :%d, max_step: %d" % (attack_user_num, max_step), flush=True)

        tree_width = 2  # Binary tree

        tree_depth = int(tree_depth)
        print("tree_depth: ", tree_depth, flush=True)


        # """ DNN
        self.act_u_history = tf.placeholder(tf.int32, [batch_size, step, tree_depth], name="actor_history") # [B, T, D] # Mark the current position to the right or left 
        self.act_u_history_flag = tf.placeholder(tf.float32, [batch_size, step, tree_depth], name="actor_history_flag") # [B, T, D]   # Whether the mark needs to be calculated, whether it is a node of the target item cluster
        self.act_old_probs = tf.placeholder(tf.float32, [batch_size, step, tree_depth], name="act_old_probs") # [B, T, D]
        self.act_rewards = tf.placeholder(tf.float32, [batch_size, ], name="real_rewards")  # [B]

        H = tf.get_variable('Tree_node_weights', [15 + int(pow(2,tree_depth)*2), hidden_size])  # target item collection + non-leaf node + leaf node

        S = tf.get_variable('Step_weights', [max_step, hidden_size])
        self.H = H
        self.S = S


        self.H_ph = tf.placeholder(tf.float32, [15 + int(pow(2,tree_depth)*2), hidden_size], name="H_ph")  # [B, T, D]
        self.H_update_op = tf.assign(H, self.H_ph)

        def get_node_emb(id):
            return tf.nn.embedding_lookup(H, id)
        # """

        with tf.variable_scope("decoder"):
            first_decoder_input = tf.tile(tf.Variable(tf.random_normal([self.attack_user_number, hidden_size]), name='first_decoder_input'),
                                          [batch_size//self.attack_user_number, 1])


            decoder_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
            zero_state = tf.zeros(shape=(batch_size, hidden_size))

            with tf.variable_scope("attention_weights", reuse=True):
                W_ref = tf.Variable(tf.random_normal([hidden_size, hidden_size], stddev=0.1), name='W_ref')

            # Training chain
            paths_loss_sum = 0
            decoder_input = first_decoder_input
            decoder_state = LSTMStateTuple(zero_state, zero_state)
            decoder_inputs = []

            for t in range(max_step):
                dec_cell_output, decoder_state = decoder_cell(inputs=decoder_input, state=decoder_state)
                level_loss = 0.0

                target_V = tf.constant(15, shape=[batch_size], name='target_V', dtype=tf.int32)  # [15, 15, 15, ...]
                decoder_first_level_V = tf.zeros(shape=[batch_size], dtype=tf.int32)  # [0, 0, 0, ...]
                decoder_other_level_V = tf.zeros(shape=[batch_size], dtype=tf.int32)
                for h in range(tree_depth):
                    if h==0:
                        level_H = tf.concat(
                            [tf.reshape(H[0, :], [1, 1, hidden_size]), tf.reshape(H[15, :], [1, 1, hidden_size])],
                            axis=1)  # 0 position and 15 position    # [1, 2, H]
                        level_H = tf.tile(level_H, [batch_size, 1, 1])  #
                    else:
                        # Based on the index of the decoder_other_level_V of the previous layer, find the embedding of the two child nodes corresponding to the next layer
                        base_V = tf.constant(int(math.pow(2, h)) - 1, dtype=tf.int32)
                        level_H = tf.concat([
                            tf.reshape(get_node_emb(decoder_first_level_V + base_V + tf.constant(2, dtype=tf.int32) * (decoder_other_level_V - tf.constant(int(math.pow(2, h - 1)) - 1, dtype=tf.int32)) + tf.constant(0, dtype=tf.int32)), shape=[batch_size, 1, hidden_size]),  # [B,H]
                            tf.reshape(get_node_emb(decoder_first_level_V + base_V + tf.constant(2, dtype=tf.int32) * (decoder_other_level_V - tf.constant(int(math.pow(2, h - 1)) - 1, dtype=tf.int32)) + tf.constant(1, dtype=tf.int32)), shape=[batch_size, 1, hidden_size])
                        ],axis=1)  # # [B, 2, H]

                    scores, score_attn = select_son_nodes(dec_cell_output, W_ref, level_H, hidden_size) # [B,2]

                    loss = tf.nn.softmax_cross_entropy_with_logits(
                        labels=tf.one_hot(self.act_u_history[:, t, h], depth=tree_width),
                        logits=scores)
                    loss *= self.act_u_history_flag[:, t, h]    # 0 or 1 is the mark really gone

                    score_attn = tf.reshape(score_attn, shape=[-1, 1])  # [B,2] -> [B*2]
                    action = self.act_u_history[:, t, h] + tf.constant([i * tree_width for i in range(batch_size)])
                    action_probs = tf.nn.embedding_lookup(score_attn, action)

                    ratio = action_probs / self.act_old_probs[:, t, h]  # [B]
                    # clip loss for PPO
                    clip_ratio = tf.minimum(ratio, tf.clip_by_value(ratio, 0.9, 1.1))

                    level_loss += (loss * clip_ratio)   # [B]


                    action = self.act_u_history[:, t, h]
                    if h==0:    # The first layer, decide whether the subsequent position starts from 15
                        decoder_first_level_V = tf.multiply(target_V, action)
                    elif h==1:
                        decoder_other_level_V = tf.constant(1, shape=[batch_size]) + action  # [B] - [B]
                    else:
                        base_V = tf.constant(int(math.pow(2, h)) - 1, dtype=tf.int32)
                        decoder_other_level_V = base_V + tf.constant(2, dtype=tf.int32)*(decoder_other_level_V-tf.constant(int(math.pow(2, h-1))-1, dtype=tf.int32)) + action    # [B] + [B] + [B]

                decoder_inputs.append(decoder_input)
                decoder_input = get_node_emb(decoder_first_level_V + decoder_other_level_V)  # [B,H]

                paths_loss_sum += level_loss    # [B]

            with tf.variable_scope("optimization"):
                self.loss = tf.reduce_mean(paths_loss_sum * self.act_rewards)  # [B] * [B]
                self.train_op = tf.train.AdamOptimizer(2e-3).minimize(self.loss)


            # Inference chain for soft
            with tf.variable_scope("inference_soft"):
                decoder_input_soft = first_decoder_input
                decoder_state_soft = LSTMStateTuple(zero_state, zero_state)
                decoder_outputs = []
                decoder_output_scores = []
                decoder_output_scores_attn = []

                for t in range(max_step):
                    dec_cell_output_soft, decoder_state_soft = decoder_cell(inputs=decoder_input_soft, state=decoder_state_soft)

                    target_V = tf.constant(15, shape=[batch_size], name='target_V', dtype=tf.int32)  # [15, 15, 15, ...]
                    decoder_first_level_V = tf.zeros(shape=[batch_size], dtype=tf.int32)  # [0, 0, 0, ...]
                    decoder_other_level_V = tf.zeros(shape=[batch_size], dtype=tf.int32)
                    decoder_layer_outputs = []
                    decoder_layer_output_scores = []
                    decoder_layer_output_scores_attn = []

                    for h in range(tree_depth):
                        if h == 0:
                            level_H = tf.concat(
                                [tf.reshape(H[0, :], [1, 1, hidden_size]), tf.reshape(H[15, :], [1, 1, hidden_size])],
                                axis=1)  # 0 position and 15 position concat up    # [1, 2, H]
                            level_H = tf.tile(level_H, [batch_size, 1, 1])  #
                        else:
                            base_V = tf.constant(int(math.pow(2, h)) - 1, dtype=tf.int32)
                            level_H = tf.concat([
                                tf.reshape(get_node_emb(
                                    decoder_first_level_V + base_V + tf.constant(2, dtype=tf.int32) * (decoder_other_level_V - tf.constant(int(math.pow(2, h - 1)) - 1, dtype=tf.int32)) + tf.constant(0,
                                                                                                       dtype=tf.int32)),shape=[batch_size, 1, hidden_size]),  # [B,H]
                                tf.reshape(get_node_emb(
                                    decoder_first_level_V + base_V + tf.constant(2, dtype=tf.int32) * (decoder_other_level_V - tf.constant(int(math.pow(2, h - 1)) - 1, dtype=tf.int32)) + tf.constant(1,
                                                                                                       dtype=tf.int32)),
                                           shape=[batch_size, 1, hidden_size])
                            ], axis=1)  # # [B, 2, H]

                        scores, attn_mask_soft = select_son_nodes(dec_cell_output_soft, W_ref, level_H, hidden_size)  # [B,2]
                        action = tf.reshape(tf.multinomial(logits=tf.log(attn_mask_soft), num_samples=1), [-1])    # [B,2] -> [B]
                        action = tf.cast(action, dtype=tf.int32)


                        if h == 0:
                            decoder_first_level_V = tf.multiply(target_V, action)
                        elif h == 1:
                            decoder_other_level_V = tf.constant(1, shape=[batch_size]) + action  # [B] - [B]    # level 1
                        else:
                            base_V = tf.constant(int(math.pow(2, h)) - 1, dtype=tf.int32)
                            decoder_other_level_V = base_V + tf.constant(2, dtype=tf.int32) * (
                            decoder_other_level_V - tf.constant(int(math.pow(2, h - 1)) - 1,
                                                                dtype=tf.int32)) + action  # [B] + [B] + [B]

                        decoder_layer_outputs.append(action)
                        decoder_layer_output_scores.append(scores)
                        decoder_layer_output_scores_attn.append(attn_mask_soft)

                    decoder_outputs.append(decoder_layer_outputs)   # [T,D,B]
                    decoder_output_scores.append(decoder_layer_output_scores)   # [T,D,B,2]
                    decoder_output_scores_attn.append(decoder_layer_output_scores_attn) # [T,D,B,2]

                    decoder_input_soft = get_node_emb(decoder_first_level_V + decoder_other_level_V)  # [B,H]


                self.decoder_outputs = tf.transpose(decoder_outputs, [2,0,1])   # [B,5,12]
                self.decoder_output_scores = tf.transpose(decoder_output_scores, [2,0,1,3])
                self.decoder_output_scores_attn = tf.transpose(decoder_output_scores_attn, [2,0,1,3]) # [B,5,12,2]


    def update_H(self, sess, H):
        sess.run([self.H_update_op], feed_dict={
            self.H_ph: H,
        })
        print("update H finished!", flush=True)



    def train_step(self, sess, pre_lists, pre_lists_flag, pre_old_probs, normd_rewards): # pre_lists_nl, pre_lists_steps, act_u_next, normd_rewards):
        tot_loss = 0.0
        for i in range(3):
            print("train i")
            _, loss = sess.run([self.train_op, self.loss], feed_dict={
                self.act_u_history: pre_lists,
                self.act_u_history_flag: pre_lists_flag,
                self.act_old_probs: pre_old_probs,
                self.act_rewards: normd_rewards,
            })
            tot_loss += loss

        return tot_loss/10.0

    def get_predicted_output(self, sess): #, pre_lists): #, pre_lists_nl, pre_lists_steps):
        pre_lists, pre_lists_max, scores, scores_mask, H, S = sess.run(
            [self.decoder_outputs, self.decoder_outputs, self.decoder_output_scores,
             self.decoder_output_scores_attn, self.H, self.S], feed_dict={
            })  # [0]

        return np.array(pre_lists), np.array(pre_lists_max), scores, scores_mask, H, S  # [B]
