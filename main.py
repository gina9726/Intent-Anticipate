# coding: utf-8

import tensorflow as tf
import argparse
import sys
import os
import numpy as np
from util import discount_cumsum
from model import RNN
from parseSeq.parse_sequence import *
import pdb

#np.random.seed(0)
#tf.set_random_seed(0)

class PolicyOptimizer(object):
    def __init__(self, task, batch_size, dropout_rate):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.RNN_model = RNN(
                dim_feat=dim_feat,
                dim_image=dim_image,
                dim_sensor=dim_sensor,
                n_words=len(vocab2idx),
                dim_hidden=dim_hidden,
                batch_size=batch_size,
                n_feat_steps=n_feat_step,
                n_label_steps=n_label_step,
                dropout_rate = dropout_rate,
                policy_hidden_dim = policy_hidden_dim,
                policy_out_dim = policy_out_dim,
                plus_rewards = plus_rewards,
                minus_rewards = minus_rewards,
                optimizer_RNN = optimizer_RNN,
                optimizer_policy = optimizer_policy,
                session = sess)
        if task == 'train':
            self.RNN_model.build_model()
        elif task == 'test':
            self.RNN_model.build_generator()

    def sample_path(self, batch_imageFT, batch_sensorFT, batch_feat_mask, batch_label):
        # Sample path from RNN model
        probs, actions, rewards, final_rewards, correct = self.RNN_model.sample(batch_imageFT, batch_sensorFT, batch_feat_mask, batch_label)

        return dict(
            actions=actions,
            rewards=rewards,
            probs=probs,
            correct=correct
        )

    def process_paths(self, paths, batch_actions_mask, batch_label):
        advantages = np.zeros(paths["rewards"].shape)
        for p in range(len(paths["rewards"])):
            rewards = paths["rewards"][p]
            actions = paths["actions"][p]
            probs = paths["probs"][p]
            probs = probs[:, int(batch_label[p][0])]
            idx = np.where(batch_actions_mask[p])
            rewards = rewards[idx]
            actions = actions[idx]
            probs = probs[idx]
            rewards = rewards*probs
            ratio = actions.sum()/float(idx[0][-1] + 1)
            if rewards[0] > 0 and ratio > 0:
                advantages[p][0:idx[0][-1]+1] = rewards*(1 - ratio)
            elif ratio > 0:
                advantages[p][0:idx[0][-1]+1] = rewards*(ratio)
        
        return dict(
            actions=paths["actions"],
            rewards=paths["rewards"],
            correct=paths["correct"],
            advantages=advantages
        )

    def train(self, model_path = 'model/combFT_2_w_nrm_256_cat'):
        global train_feat, train_label
        
        with tf.device('/cpu:0'):
            saver = tf.train.Saver(tf.all_variables())

        dataLen = len(train_feat)
        feat_mask = np.zeros([dataLen, n_feat_step])
        iteration = 1
        exp = lambda n, x: np.exp(np.log(0.1)/n*(n-x))

        for epoch in range(1, n_epochs):
            #shuffle data
            index = np.random.permutation(range(len(train_feat)))
            train_feat = train_feat[index]
            train_label = train_label[index]
            print 'epoch:', epoch
            #batch
            for start,end in zip(
                    range(0, dataLen, batch_size),
                    range(batch_size, dataLen+1, batch_size)):
                
                batch_imageFT = np.zeros([batch_size, n_feat_step, dim_image])
                batch_sensorFT = np.zeros([batch_size, n_feat_step, dim_sensor])
                batch_feat_mask = np.zeros([batch_size, n_feat_step])
                batch_loss_mask = np.zeros([batch_size, n_feat_step])
                batch_actions_mask = np.zeros([batch_size, n_feat_step], dtype=bool)
                batch_label = np.zeros([batch_size, n_label_step])

                current_feat = train_feat[start:end]
                current_feat_mask = feat_mask[start:end]
                current_label = train_label[start:end]

                for b in xrange(batch_size):
                    idx = len(current_feat[b])
                    batch_sensorFT[b][0:idx] = np.concatenate([current_feat[b][:,0:512], current_feat[b][:,2560:2560+512]],axis=1) 
                    batch_imageFT[b][0:idx] = np.concatenate([current_feat[b][:,512:2560], current_feat[b][:,2560+512:]],axis=1) 
                    batch_feat_mask[b][idx-1] = 1
                    batch_loss_mask[b][0:idx-1] = exp(np.ones([idx-1])*(idx-1), range(idx-1))
                    batch_actions_mask[b][0:idx-1] = True # should mask before last step
                    batch_actions_mask[b][0] = True # should mask before last step
                    idx = int(len(current_label[b]))
                    batch_label[b][0:idx] = current_label[b]
                
                paths = self.sample_path(batch_imageFT, batch_sensorFT, batch_feat_mask, batch_label)
                data = self.process_paths(paths, batch_actions_mask, batch_label)
                loss, sloss, closs = self.RNN_model.train(batch_imageFT, batch_sensorFT, batch_feat_mask, batch_loss_mask, batch_label, data["actions"], data["advantages"], batch_actions_mask)
                # Result
                avg_return = 0.0
                avg_action = 0.0
                for b in xrange(batch_size):
                    avg_return += data["advantages"][b][0]
                    idx = np.where(batch_actions_mask[b])
                    actions = data["actions"][b][idx]
                    actions = np.concatenate([[0], actions]) 
                    action_ratio = actions.mean()
                    avg_action += action_ratio
                avg_return /= batch_size 
                avg_action /= batch_size 
                avg_accuracy = data["correct"].mean()

                print("Epoch {} Iteration {}: Average Return = {}".format(epoch, iteration, avg_return))
                print("Epoch {} Iteration {}: Action Ratio = {}".format(epoch, iteration, avg_action))
                print("Epoch {} Iteration {}: Total Loss = {}".format(epoch, iteration, loss))
                print("Epoch {} Iteration {}: Surrogate Loss = {}".format(epoch, iteration, sloss))
                print("Epoch {} Iteration {}: Cross Entropy Loss = {}".format(epoch, iteration, closs))
                print("Epoch {} Iteration {}: Accuracy = {}".format(epoch, iteration, avg_accuracy))
                iteration += 1

            if np.mod(epoch, 100) == 0:
                print "Epoch ", epoch, " is done. Saving the model ..."
                with tf.device('/cpu:0'):
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                    saver.save(sess, os.path.join(model_path, 'RL-RNN'), global_step=epoch)

    def test(self):

        global test_feat, test_label
        total = 0
        corrects = 0
        avg_action = 0.0
        action_thresh = args.thresh
        for i in xrange(len(test_feat)):
            image = np.zeros([1, n_feat_step, dim_image])
            sensor = np.zeros([1, n_feat_step, dim_sensor])
            feat_mask = np.zeros([1, n_feat_step])
            label = np.zeros([1, n_label_step])
            idx = len(test_feat[i])
            sensor[0][0:idx] = np.concatenate([test_feat[i][:,0:512], test_feat[i][:,2560:2560+512]],axis=1)
            image[0][0:idx] = np.concatenate([test_feat[i][:,512:2560], test_feat[i][:,2560+512:]],axis=1) 
            feat_mask[0][idx-1] = 1
            idx = int(len(test_label[i]))
            label[0][0:idx] = test_label[i]

            probs_buff, actions, rewards, final_rewards, correct, predict = self.RNN_model.test(image, sensor, feat_mask, label, action_thresh)

            idx = np.where(feat_mask == 1) # index for video length
            ##default motion
            action_ratio = (actions[idx[0],:idx[1]].sum())/(idx[1][0] + 1) 
            if np.isnan(action_ratio):
                action_ratio = 0
            avg_action += action_ratio
            #calculate accuracy
            label_intention = label[0][-1]
            predict_intention = predict[0][-1]
            if correct[0][0]:
                corrects += 1
            total += 1
            print '-----------------------------------------'
            print 'Test sequence: ', total 
            print 'label intention:', idx2vocab[label_intention]
            print 'predict intention:', idx2vocab[predict_intention]
            print "current action ratio: ", action_ratio
            print "average action ratio: ", avg_action/float(total)
            print "current accuracy =", (corrects/float(total))*100
            print '-----------------------------------------'

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='RL-RNN model')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--task', dest='task',
                        help='train or test',
                        default='train', type=str)
    parser.add_argument('--net', dest='model',
                        help='model for training or testing',
                        default='combFT_2_w_nrm_256_cat', type=str)
    parser.add_argument('--feat', dest='feat',
                        help='select feature directory',
                        default='Res50_nocalib_w_nrm', type=str)
    parser.add_argument('--percent', dest='percent',
                        help='percentage of input lenght',
                        default=1.0, type=float)
    parser.add_argument('--iname', dest='iname',
                        help='input feature name',
                        default='combfeat6F', type=str)
    parser.add_argument('--trainR', dest='trainR',
                        help='training ratio',
                        default=0.5, type=float)
    parser.add_argument('--orderR', dest='orderR',
                        help='reduce order ratio',
                        default=1, type=float)
    parser.add_argument('--user', dest='user',
                        help='user',
                        default='C', type=str)
    parser.add_argument('--plus_reward', dest='plus_reward',
                        help='plus reward',
                        default=100, type=int)
    parser.add_argument('--minus_reward', dest='minus_reward',
                        help='minus reward',
                        default=-100, type=int)
    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory for saving model',
                        default='', type=str)
    parser.add_argument('--thresh', dest='thresh',
                        help='set threshold to trigger action',
                        default=0.5, type=float)
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

global args
args = parse_args()

def get_data(data_path):
    global dim_feat, n_feat_step, n_label_step
    train_feat = []
    test_feat = []
    train_label = []
    test_label = []
    train_path = []
    test_path = []
    maxLen_feat = 0
    maxLen_label = 0
    dataLen = 0
    print "reading data, percent: ", args.percent
    print "train_ratio : ", args.trainR
    for seqDir in os.listdir(data_path):            
        idx = int(seqDir.lstrip('seq_'))
        tmp_feat = []
        tmp_label = []
        tmp_path = []
        I = len(os.listdir(os.path.join(data_path, seqDir)))
        for insIdx in range(1, I+1):
            insDir = 'I' + str(insIdx)
            modelDir = os.path.join(data_path, seqDir, insDir)
            modelNames = os.listdir(modelDir)
            for modelName in modelNames:
                if modelName == args.feat:
                    fileDir = os.path.join(modelDir, modelName)
                    fileR1 = os.path.join(fileDir,args.iname+"_right.npy")
                    featR = np.load(fileR1)
                    fileL1 = os.path.join(fileDir,args.iname+"_left.npy")
                    featL = np.load(fileL1)
                    filePath = os.path.join(fileDir, fileR1)
                    Table=parse_order(args.orderR)
                    reduce_order=seqid2int[:,0]*Table
                    if reduce_order[idx] != 0:
                        tmp_feat.append(np.concatenate((featR, featL), axis=1)) 
                        tmp_label.append(seqid2int[idx]) #intention only

                        if args.percent != 1:
                            x = int(np.ceil(len(tmp_feat[-1])*args.percent))
                            tmp_feat[-1] = tmp_feat[-1][0:x]
                        
                        tmp_path.append(filePath)
                        featR = []
                        featL = []
                        if len(tmp_feat[-1]) > maxLen_feat:
                            maxLen_feat = len(tmp_feat[-1])
                        if len(tmp_label[-1]) > maxLen_label:
                            maxLen_label = len(tmp_label[-1])
                        dataLen += 1
                        dim_feat = tmp_feat[-1].shape[1]
        
        x = int(np.floor(args.trainR*I))
        train_feat.extend(tmp_feat[0:x])
        train_label.extend(tmp_label[0:x])
        train_path.extend(tmp_path[0:x])
        if args.task == 'test_train':
            test_feat.extend(tmp_feat[0:x])
            test_label.extend(tmp_label[0:x])
            test_path.extend(tmp_path[0:x])
        elif args.task == 'valid': 
            test_feat.extend(tmp_feat[1:I-1])
            test_label.extend(tmp_label[1:I-1])
            test_path.extend(tmp_path[1:I-1])
        else:
            test_feat.extend(tmp_feat[x:])
            test_label.extend(tmp_label[x:])
            test_path.extend(tmp_path[x:])
            
    n_feat_step = maxLen_feat
    n_label_step = maxLen_label
    
    for p in train_path:
        print 'train path: ', p

    for p in test_path:
        print 'test path: ', p
    return np.array(train_feat), np.array(train_label), np.array(test_feat), np.array(test_label)

############## Train Parameters #################
#RNN
dim_hidden = 256
dim_feat = 5120
dim_image = 4096
dim_sensor = 1024
n_feat_step = 0
n_label_step = 0
n_epochs = 501
batch_size = 40
#Policy
learning_rate = 0.001
policy_hidden_dim = 200
policy_out_dim = 2
plus_rewards = [0, args.plus_reward] 
minus_rewards = [0, args.minus_reward] 
##################################################

data_path='/home/james/Desktop/extract_intention_feat/NewMean/feats_' + args.user
# Load data
train_feat, train_label, test_feat, test_label = get_data(data_path)
print '# of training data: ', len(train_feat)
print '# of testing data: ', len(test_feat)
print args

# Session
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
optimizer_RNN = tf.train.AdamOptimizer(learning_rate=learning_rate/10)
optimizer_policy = tf.train.AdamOptimizer(learning_rate=learning_rate)
if args.task == 'train':
    # Train the policy optimizer
    po = PolicyOptimizer(args.task, batch_size, 0.5)
    
    sess.run(tf.initialize_all_variables())
    t_vars = tf.trainable_variables()
    RNN_vars = po.RNN_model.RNN_vars
    policy_vars = po.RNN_model.policy_vars
    with tf.device('/cpu:0'):
        saver = tf.train.Saver(RNN_vars)
        saver.restore(sess, args.model)

    po.train(args.save_dir)
elif args.task == 'test' or args.task == 'valid' or args.task == 'test_train':
    # Test the policy optimizer
    po = PolicyOptimizer('test', 1, 0)

    sess.run(tf.initialize_all_variables())
    t_vars = po.RNN_model.t_vars
    with tf.device('/cpu:0'):
        saver = tf.train.Saver(t_vars)
        saver.restore(sess, args.model)
    po.test()
