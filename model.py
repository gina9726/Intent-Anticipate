#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pdb
import cv2

rnn_cell = tf.nn.rnn_cell

def print_tf(t):
    print(t.op.name, ' ', t.get_shape().as_list())

class RNN():
    def __init__(self, dim_feat, dim_image, dim_sensor, n_words, dim_hidden, batch_size, n_feat_steps, n_label_steps, dropout_rate, policy_hidden_dim, policy_out_dim, plus_rewards, minus_rewards, optimizer_RNN, optimizer_policy, session):
        self.dim_feat = dim_feat
        self.dim_image = dim_image
        self.dim_sensor = dim_sensor
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_feat_steps = n_feat_steps
        self.n_label_steps = n_label_steps
        self.dropout_rate = dropout_rate
        self.policy_hidden_dim = policy_hidden_dim
        self.policy_out_dim = policy_out_dim
        self.plus_rewards = plus_rewards 
        self.minus_rewards = minus_rewards
        self._opt_RNN = optimizer_RNN
        self._opt_policy = optimizer_policy
        self._sess = session
        self.probs_buff = []

        # LSTM
        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')

        self.lstm1 = rnn_cell.LSTMCell(self.dim_hidden, use_peepholes = True, initializer=tf.random_uniform_initializer(-0.1,0.1))
        self.lstm1_dropout = rnn_cell.DropoutWrapper(self.lstm1,output_keep_prob=1 - self.dropout_rate)
        self.lstm2 = rnn_cell.LSTMCell(self.dim_hidden, use_peepholes = True, initializer=tf.random_uniform_initializer(-0.1,0.1))
        self.lstm2_dropout = rnn_cell.DropoutWrapper(self.lstm2,output_keep_prob=1 - self.dropout_rate)

        # Embed combine feature
        self.encode_image_W = tf.Variable( tf.random_uniform([dim_feat, dim_hidden], -0.1, 0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_image_b')

        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1), name='embed_word_W')
        self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

        # Placeholder Inputs
        self.video = tf.placeholder(tf.float32, [self.batch_size, self.n_feat_steps, self.dim_feat])
        self.image = tf.placeholder(tf.float32, [self.batch_size, self.n_feat_steps, self.dim_image])
        self.sensor = tf.placeholder(tf.float32, [self.batch_size, self.n_feat_steps, self.dim_sensor])
        self.video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_feat_steps])
        self.loss_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_feat_steps])
        self.label = tf.placeholder(tf.int32, [self.batch_size, self.n_label_steps])

        self.policy_input_dim = self.lstm1.state_size+self.lstm2.state_size+self.dim_hidden
        self.is_Train = tf.placeholder(tf.bool, [self.batch_size, 1], name="is_Train")
        self._actions = tf.placeholder(tf.int32, [self.batch_size, self.n_feat_steps], name="actions")
        self._advantages = tf.placeholder(tf.float32, [self.batch_size, self.n_feat_steps], name="advantages")
        self._actions_mask = tf.placeholder(tf.bool, [self.batch_size, self.n_feat_steps], name="actions_mask")
        self._action_thresh = tf.placeholder(tf.float32, [1], name="action_thresh")

        # Policy Network 
        self.W1 = tf.Variable(tf.random_normal([self.policy_input_dim, policy_hidden_dim]), name="policy_W1")
        self.b1 = tf.Variable(tf.zeros([policy_hidden_dim]), name="policy_b1")
        self.W2 = tf.Variable(tf.random_normal([policy_hidden_dim, policy_out_dim]), name="policy_W2")
        self.b2 = tf.Variable(tf.zeros([policy_out_dim]), name="policy_b2")

    def build_model(self):

        tmp_image = tf.zeros([self.batch_size, self.dim_image])
        selected_mask = tf.zeros([self.batch_size, 1]) # 1 :combine feature, 0 :sensor feature
        state1 = tf.zeros([self.batch_size, self.lstm1.state_size]) #100x1000
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size]) #100x1000
        padding = tf.zeros([self.batch_size, self.dim_hidden])
        deadzone_lock = tf.zeros([self.batch_size, 1])
        cross_entropys = []
        output_probs = []
        state1_tmp = []
        state2_tmp = []
        actions = []
        rewards = []

        ## Encoder 
        for i in range(self.n_feat_steps): 
            # Update tmp image
            image_mask = tf.tile(selected_mask, tf.constant([1, self.dim_image]))
            tmp_image = tf.mul(self.image[:,i,:], image_mask) + tf.mul(tmp_image, tf.sub(tf.ones_like(image_mask), image_mask))
            tmp_sensor = self.sensor[:,i,:]

            # Combine sensor and image feature
            tmp_comb = tf.concat(1, [tmp_sensor[:,0:512], tmp_image[:,0:2048], tmp_sensor[:,512:], tmp_image[:, 2048:]])

            # Embed combined feature
            video_flat = tf.reshape(tmp_comb, [-1, self.dim_feat]) # (batch_size*n_lstm_steps) x dim_feat
            video_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b) # (batch_size*n_lstm_steps, dim_hidden)
            video_emb = tf.reshape(video_emb, [self.batch_size, self.dim_hidden])

            if i > 0:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1_dropout( video_emb, state1 ) # def __call__(self, inputs, state, scope=None)
                
            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2_dropout( tf.concat(1,[padding, output1]), state2 )
        
            state1_tmp.append(state1)
            state2_tmp.append(state2)

            # Take action
            state_concat = tf.concat(1, [state1, state2, video_emb]) # concatenate state
            action_probs = self.act(state_concat)
            self.probs_buff.append(action_probs)

            selected_mask = tf.cast(tf.select(self.is_Train, tf.expand_dims(self._actions[:,i], 1), tf.cast(tf.multinomial(action_probs, 1), tf.int32)), tf.float32)
                        
            # Save sample path
            actions.append(selected_mask)
        
        self.actions = tf.pack(actions)
        self.actions = tf.transpose(self.actions, perm=[1, 0, 2])
        self.actions = tf.reduce_sum(self.actions, 2)
        
        ## Decoder
        current_embed = tf.zeros([self.batch_size, self.dim_hidden])

        ### label to one hot vector ###
        indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)    # Same size as label
        concated = tf.concat(1, [indices, self.label])  # batch_size x 2
        
        with tf.device('/cpu:0'):
            onehot_labels = tf.sparse_to_dense(concated, tf.pack([self.batch_size, self.n_words]), 1.0, 0.0)
        ### End of onehot ###

        tf.get_variable_scope().reuse_variables()
        for i in range(self.n_feat_steps):
            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1_dropout( padding, state1_tmp[i] )

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2_dropout( tf.concat(1,[current_embed, output1]), state2_tmp[i] )

            # cross entropy loss
            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_words, onehot_labels)
        
            cross_entropys.append(cross_entropy)
            output_probs.append(tf.nn.softmax(logit_words))
        
        mask = tf.to_float(self.video_mask)
        mask = tf.expand_dims(mask, 2)
        mask = tf.tile(mask, tf.constant([1, 1, self.n_words]))  # batch_size x n_feat_steps x 1

        output_probs = tf.pack(output_probs)
        self.output_probs = tf.transpose(output_probs, perm=[1, 0, 2]) # batch_size x n_feat_steps x self.n_words
        last_logit_words = tf.reduce_sum(tf.mul(self.output_probs, mask), 1)

        cross_entropys = tf.pack(cross_entropys)
        cross_entropys = tf.transpose(cross_entropys, perm=[1, 0]) # batch_size x n_feat_steps
        cross_entropys = tf.mul(cross_entropys, self.loss_mask)
        video_length = tf.argmax(self.video_mask, 1) + 1

        self.cross_entropy_loss = tf.reduce_mean(tf.div(tf.reduce_sum(cross_entropys, 1), tf.cast(video_length, tf.float32)))

        # Correctness
        with tf.device('/cpu:0'):
            self.max_prob_index = tf.expand_dims(tf.argmax(last_logit_words, 1), 1)
        self.correct = tf.equal(tf.cast(self.max_prob_index, tf.int32), self.label)
        self.final_rewards = tf.select(self.correct, tf.ones_like(self.label)*self.plus_rewards[1], tf.ones_like(self.label)*self.minus_rewards[1])
        self.rewards = tf.tile(self.final_rewards, tf.constant([1, self.n_feat_steps]))
        
        # Surroate Loss
        probs = tf.pack(self.probs_buff)
        probs = tf.transpose(probs, perm=[1, 0, 2]) # batch_size*n_feat_steps*policy_out_dim
        probs_shape = probs.get_shape().as_list()
        action_idxs_flattened = tf.range(0, probs_shape[0]*probs_shape[1]) * probs_shape[2]
        action_idxs_flattened += tf.reshape(self._actions, [-1]) # batch_size*n_feat_steps
        probs_vec = tf.gather(tf.reshape(probs, [-1]), action_idxs_flattened)
        log_prob = tf.log(probs_vec + 1e-8)
        tmp_loss = tf.mul(log_prob, tf.reshape(self._advantages, [-1]))
        #Mask out actions that don't care
        self.action_idxs_mask = tf.where(tf.reshape(self._actions_mask, [-1]))
        tmp_loss = tf.gather(tmp_loss, self.action_idxs_mask)

        self.surr_loss = -tf.reduce_mean(tmp_loss)

        self.total_loss = self.cross_entropy_loss + self.surr_loss

        t_vars = tf.trainable_variables()
        self.policy_vars = [var for var in t_vars if 'policy' in var.name]
        self.RNN_vars = [var for var in t_vars if 'policy' not in var.name]
        
        grads_and_vars = self._opt_RNN.compute_gradients(self.cross_entropy_loss, self.RNN_vars)
        self.train_op_RNN = self._opt_RNN.apply_gradients(grads_and_vars, name="train_op_RNN")

        grads_and_vars_policy = self._opt_policy.compute_gradients(self.surr_loss, self.policy_vars)
        self.train_op_policy = self._opt_policy.apply_gradients(grads_and_vars_policy, name="train_op_policy")
    
    def build_generator(self):

        selected_mask = tf.zeros([self.batch_size, 1]) # 1 :combine feature, 0 :sensor feature
        tmp_image = tf.zeros([self.batch_size, self.dim_image])
        state1 = tf.zeros([self.batch_size, self.lstm1.state_size]) # 100x1000
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size]) # 100x1000
        padding = tf.zeros([self.batch_size, self.dim_hidden])
        output_probs = []
        state1_tmp = []
        state2_tmp = []
        actions = []
        rewards = []

        ## Encoder 
        for i in range(self.n_feat_steps): 
            # Update tmp image
            image_mask = tf.tile(selected_mask, tf.constant([1, self.dim_image]))
            tmp_image = tf.mul(self.image[:,i,:], image_mask) + tf.mul(tmp_image, tf.sub(tf.ones_like(image_mask), image_mask))
            tmp_sensor = self.sensor[:,i,:]

            # Combine sensor and image feature
            tmp_comb = tf.concat(1, [tmp_sensor[:,0:512], tmp_image[:,0:2048], tmp_sensor[:,512:], tmp_image[:, 2048:]])

            # Embed combined feature
            video_flat = tf.reshape(tmp_comb, [-1, self.dim_feat]) # (batch_size*n_lstm_steps) x dim_feat
            video_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b) # (batch_size*n_lstm_steps, dim_hidden)
            video_emb = tf.reshape(video_emb, [self.batch_size, self.dim_hidden])

            if i > 0:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1_dropout( video_emb, state1 ) # def __call__(self, inputs, state, scope=None)
                
            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2_dropout( tf.concat(1,[padding, output1]), state2 )
        
            state1_tmp.append(state1)
            state2_tmp.append(state2)

            # Take action
            state_concat = tf.concat(1, [state1, state2, video_emb]) # concatenate state
            action_probs = self.act(state_concat)
            self.probs_buff.append(action_probs)
            selected_mask = tf.cast(tf.expand_dims(tf.greater(action_probs[:,1], self._action_thresh), 1), tf.float32)

            # Save sample path
            actions.append(selected_mask)

        print_tf(output1)
        print_tf(output2)
        print_tf(state1)
        print_tf(state2)
        
        self.actions = tf.pack(actions)
        self.actions = tf.transpose(self.actions, perm=[1, 0, 2])
        self.actions = tf.reduce_sum(self.actions, 2)
        
        state1_tmp = tf.pack(state1_tmp)    # MaxFeatLen x 100 x 1000
        mask = tf.to_float(self.video_mask)
        mask = tf.expand_dims(mask, 2)
        mask = tf.tile(mask, tf.constant([1, 1, self.lstm1.state_size]))
        mask = tf.transpose(mask, perm=[1, 0, 2])
        state1 = tf.reduce_sum(tf.mul(state1_tmp, mask), 0)
        state2_tmp = tf.pack(state2_tmp)
        mask = tf.to_float(self.video_mask)
        mask = tf.expand_dims(mask, 2)
        mask = tf.tile(mask, tf.constant([1, 1, self.lstm2.state_size]))
        mask = tf.transpose(mask, perm=[1, 0, 2])
        state2 = tf.reduce_sum(tf.mul(state2_tmp, mask), 0)

        ## Decoder 
        current_embed = tf.zeros([self.batch_size, self.dim_hidden])
        tf.get_variable_scope().reuse_variables()
        with tf.variable_scope("LSTM1"):
            output1, state1 = self.lstm1_dropout( padding, state1 )

        with tf.variable_scope("LSTM2"):
            output2, state2 = self.lstm2_dropout( tf.concat(1,[current_embed, output1]), state2 )

        ### label to one hot vector ###
        indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)    # Same size as label 
        concated = tf.concat(1, [indices, self.label])  # batch_size x 2
        
        with tf.device('/cpu:0'):
            onehot_labels = tf.sparse_to_dense(concated, tf.pack([self.batch_size, self.n_words]), 1.0, 0.0)
        ### End of onehot ###

        logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_words, onehot_labels)

        output_probs.append(logit_words)
        self.cross_entropy_loss = tf.reduce_mean(cross_entropy)

        # Correctness
        with tf.device('/cpu:0'):
            self.max_prob_index = tf.expand_dims(tf.argmax(logit_words, 1), 1)
        self.correct = tf.equal(tf.cast(self.max_prob_index, tf.int32), self.label)
        self.final_rewards = tf.select(self.correct, tf.ones_like(self.label)*self.plus_rewards[1], tf.ones_like(self.label)*self.minus_rewards[1])
        self.rewards = tf.tile(self.final_rewards, tf.constant([1, self.n_feat_steps]))
        
        self.t_vars = tf.trainable_variables()
        
    def act(self, _observations):
        output1 = tf.nn.tanh(tf.matmul(_observations, self.W1) + self.b1)
        action_probs = tf.nn.softmax(tf.matmul(output1, self.W2) + self.b2)

        return action_probs 

    def sample(self, batch_imageFT, batch_sensorFT, batch_feat_mask, batch_label):
        probs, actions, rewards, final_rewards, correct = self._sess.run(
                [self.output_probs, self.actions, self.rewards, self.final_rewards, self.correct],
                feed_dict={
                    self.image: batch_imageFT,
                    self.sensor: batch_sensorFT,
                    self.video_mask : batch_feat_mask,
                    self.loss_mask : np.zeros([self.batch_size, self.n_feat_steps]),
                    self.label: batch_label,
                    self._actions: np.zeros([self.batch_size, self.n_feat_steps], dtype=np.int64),
                    self._advantages : np.zeros([self.batch_size, self.n_feat_steps]),
                    self._actions_mask : np.zeros([self.batch_size, self.n_feat_steps], dtype=bool),
                    self.is_Train: np.zeros([self.batch_size, 1], dtype=bool) # False
                    })
        return probs, actions, rewards, final_rewards, correct

    def train(self, batch_imageFT, batch_sensorFT, batch_feat_mask, batch_loss_mask, batch_label, actions, advantages, actions_mask):
        loss, sloss, closs, _, _ = self._sess.run(
                [self.total_loss, self.surr_loss, self.cross_entropy_loss, self.train_op_RNN, self.train_op_policy],
                feed_dict={
                    self.image: batch_imageFT,
                    self.sensor: batch_sensorFT,
                    self.video_mask : batch_feat_mask,
                    self.loss_mask : batch_loss_mask,
                    self.label: batch_label,
                    self._actions: actions,
                    self._advantages : advantages,
                    self._actions_mask : actions_mask,
                    self.is_Train: np.ones([self.batch_size, 1], dtype=bool) # True
                    })
        return loss, sloss, closs 

    def train_RNN(self, batch_imageFT, batch_sensorFT, batch_feat_mask, batch_loss_mask, batch_label, actions, advantages, actions_mask):
        loss, sloss, closs, _ = self._sess.run(
                [self.total_loss, self.surr_loss, self.cross_entropy_loss, self.train_op_RNN],
                feed_dict={
                    self.image: batch_imageFT,
                    self.sensor: batch_sensorFT,
                    self.video_mask : batch_feat_mask,
                    self.loss_mask : batch_loss_mask,
                    self.label: batch_label,
                    self._actions: actions,
                    self._advantages : advantages,
                    self._actions_mask : actions_mask,
                    self._action_thresh : [0],
                    self.is_Train: np.ones([self.batch_size, 1], dtype=bool) # True
                    })
        return loss, sloss, closs 

    def train_policy(self, batch_imageFT, batch_sensorFT, batch_feat_mask, batch_loss_mask, batch_label, actions, advantages, actions_mask):
        loss, sloss, closs, _ = self._sess.run(
                [self.total_loss, self.surr_loss, self.cross_entropy_loss, self.train_op_policy],
                feed_dict={
                    self.image: batch_imageFT,
                    self.sensor: batch_sensorFT,
                    self.video_mask : batch_feat_mask,
                    self.loss_mask : batch_loss_mask,
                    self.label: batch_label,
                    self._actions: actions,
                    self._advantages : advantages,
                    self._actions_mask : actions_mask,
                    self._action_thresh : [0],
                    self.is_Train: np.ones([self.batch_size, 1], dtype=bool) # True
                    })
        return loss, sloss, closs 

    def test(self, batch_imageFT, batch_sensorFT, batch_feat_mask, batch_label, action_thresh):
        action_probs, actions, rewards, final_rewards, correct, index = self._sess.run(
                [self.probs_buff, self.actions, self.rewards, self.final_rewards, self.correct, self.max_prob_index],
                feed_dict={
                    self.image: batch_imageFT,
                    self.sensor: batch_sensorFT,
                    self.video_mask : batch_feat_mask,
                    self.loss_mask : np.zeros([self.batch_size, self.n_feat_steps]),
                    self.label: batch_label,
                    self._actions: np.zeros([self.batch_size, self.n_feat_steps], dtype=np.int64),
                    self._advantages : np.zeros([self.batch_size, self.n_feat_steps]),
                    self._actions_mask : np.zeros([self.batch_size, self.n_feat_steps], dtype=bool),
                    self._action_thresh : [action_thresh],
                    self.is_Train: np.zeros([self.batch_size, 1], dtype=bool) # False
                    })
        return action_probs, actions, rewards, final_rewards, correct, index
