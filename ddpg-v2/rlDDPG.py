# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 14:02:53 2019

@author: Cheng_Home
"""
"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

"""

#######################  connect to the V-rep  ################################

# reset the radom seed of numpy
import datetime
import math
import tensorflow as tf
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import os
import shutil
sys.path.append('../')
import RemoteAPIs.vrep as vrep
np.random.seed(1)
# reset the radom seed of tensorflow
tf.set_random_seed(1)
# reset the static graph
tf.reset_default_graph()
LOAD = False
TEST = True
SAVEDIR = "./saveModel/"
LOGDIR = "./log/"
DIR = [SAVEDIR, LOGDIR]
for directory in DIR:
    if not os.path.exists(directory):
        os.mkdir(directory)
if TEST:
    for directory in DIR:
        if len(os.listdir(directory)) != 0:
            shutil.rmtree(directory)
            os.mkdir(directory)

# connect to the vrep
clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
if clientID != -1:
    print('Connected to remote API server')

#####################  hyper parameters  ####################

MAX_EPISODES = 15000
LR_A = 0.0001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
GAMMA = 0.9  # reward discount
TAU = 0.01  # soft replacement
MEMORY_CAPACITY = 800000
BATCH_SIZE = 256

###############################  DDPG  ####################################


class DDPG(object):
    def __init__(self,
                 a_dim,
                 s_dim,
                 a_bound,
                 outputGraph=True,
                 saveModel=True):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim * 2),
                               dtype=np.float32)
        self.pointer = 0
        self.best_r = 0

        self.sess = tf.Session()
        self.a_dim, self.s_dim, self.a_bound, self.outputGraph, self.saveModel = a_dim, s_dim, a_bound, outputGraph, saveModel
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                           scope='Critic/target')

        #  soft replacement
        self.soft_replace = [
            tf.assign(t, (1 - TAU) * t + TAU * e)
            for t, e in zip(self.at_params + self.ct_params, self.ae_params +
                            self.ce_params)
        ]

        a_loss = -tf.reduce_mean(q, name="a_loss")  # maximize the q
        with tf.variable_scope("td_error"):
            q_target = self.R + GAMMA * q_
            # in the feed_dic for the td_error, the self.a should change to actions in memory
            td_error = tf.losses.mean_squared_error(labels=q_target,
                                                    predictions=q)

        with tf.variable_scope("Train"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.atrain = tf.train.AdamOptimizer(LR_A).minimize(
                    a_loss, var_list=self.ae_params, name="atrain")
                self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(
                    td_error, var_list=self.ce_params, name="ctrain")

        #  Initailize the global variables
        self.sess.run(tf.global_variables_initializer())

        if self.saveModel:
            self.saver = tf.train.Saver(var_list=tf.trainable_variables())
        if self.outputGraph:
            tf.summary.FileWriter("log/", self.sess.graph)

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim:self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1:-self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run([self.ctrain], {
            self.S: bs,
            self.a: ba,
            self.R: br,
            self.S_: bs_
        })
        self.sess.run([self.atrain], {self.S: bs})

        # soft target replacement
        self.sess.run(self.soft_replace)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        # replace the old memory with new memory
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        init_w = tf.random_normal_initializer(0., 0.1)
        init_b = tf.constant_initializer(0.1)
        hidden1 = 128
        hidden2 = 64
        with tf.variable_scope(scope):
            layer1 = tf.layers.dense(s,
                                     hidden1,
                                     activation=None,
                                     kernel_initializer=init_w,
                                     bias_initializer=init_b,
                                     name='l1',
                                     trainable=trainable)
            layer1 = tf.layers.batch_normalization(layer1,
                                                   training=trainable,
                                                   name="norms1")
            layer2 = tf.layers.dense(layer1,
                                     hidden2,
                                     activation=None,
                                     kernel_initializer=init_w,
                                     bias_initializer=init_b,
                                     name='l2',
                                     trainable=trainable)
            layer2 = tf.layers.batch_normalization(layer2,
                                                   training=trainable,
                                                   name="norms2")
            a = tf.layers.dense(layer2,
                                self.a_dim,
                                activation=tf.nn.tanh,
                                name='a',
                                kernel_initializer=init_w,
                                bias_initializer=init_b,
                                trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        init_w = tf.random_normal_initializer(0., 0.1)
        init_b = tf.constant_initializer(0.1)
        with tf.variable_scope(scope):
            hidden1 = 256
            hidden2 = 128
            layer_s = tf.layers.dense(s,
                                      hidden1,
                                      activation=None,
                                      kernel_initializer=init_w,
                                      bias_initializer=init_b,
                                      name='l1_s',
                                      trainable=trainable)
            layer_s = tf.layers.batch_normalization(layer_s,
                                                    training=trainable,
                                                    name="norms1")
            layer_s = tf.layers.dense(layer_s,
                                      hidden2,
                                      activation=None,
                                      kernel_initializer=init_w,
                                      bias_initializer=init_b,
                                      name='l2_s',
                                      trainable=trainable)
            layer_s = tf.layers.batch_normalization(layer_s,
                                                    training=trainable,
                                                    name="norms2")
            layer_a = tf.layers.dense(a,
                                      hidden2,
                                      activation=None,
                                      name='a',
                                      trainable=trainable)
            net = tf.layers.dense(tf.keras.layers.concatenate(
                [layer_s, layer_a]),
                                  hidden2,
                                  activation=tf.nn.relu,
                                  name='cat',
                                  kernel_initializer=init_w,
                                  bias_initializer=init_b,
                                  trainable=trainable)
            net = tf.layers.dense(net,
                                  self.a_dim,
                                  activation=None,
                                  name='Q',
                                  kernel_initializer=init_w,
                                  bias_initializer=init_b,
                                  trainable=trainable)
            return net

    def save_Model(self, DIR, reward, episode):
        print("best_reward:{:+.3f}, reward:{:+.3f}".format(
            self.best_r, reward))
        if self.best_r < reward:
            print("Model save!")
            modelName = datetime.datetime.now().strftime("%y_%m_%d")
            self.saver.save(self.sess, DIR + modelName, global_step=episode)
            self.best_r = reward

    def load_Model(self, DIR, load=False):
        if load:
            if len(os.listdir(DIR)) == 0:
                print(
                    "There is not any model in the {} ! Restart trainning...\n"
                    .format(DIR))
                time.sleep(1)
                return False
            else:
                print("Loading succeed!")
                self.saver.restore(self.sess, tf.train.latest_checkpoint(DIR))
                return True
        else:
            print("Start training...\n")
            return False

    def close(self):
        self.sess.close()


class OUNoise(object):
    def __init__(self,
                 action_space=[],
                 mu=0.0,
                 theta=0.15,
                 max_sigma=0.3,
                 min_sigma=0.3,
                 decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = len(action_space)
        self.low = min(action_space[0])
        self.high = max(action_space[0])
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * \
            np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - \
            (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


###############################  function  ####################################


def clean_step():
    vrep.simxClearFloatSignal(clientID, "action", vrep.simx_opmode_blocking)
    vrep.simxClearIntegerSignal(clientID, "line", vrep.simx_opmode_blocking)


def step(action, lineNum):
    # should clean pervious step
    clean_step()
    # send the action which controls the behavier of manipulator and the line which restricts the order of each actions to vrep
    # vrep.simx_opmode means that sending the action once in one single time
    vrep.simxSetFloatSignal(clientID, "action", action,
                            vrep.simx_opmode_blocking)
    vrep.simxSetIntegerSignal(clientID, "line", lineNum,
                              vrep.simx_opmode_blocking)
    # activate the synchronous mode
    vrep.simxSynchronousTrigger(clientID)
    vrep.simxGetPingTime(clientID)
    # from the buffer retrieve the information which involves the velocity, angle, reward ,and the terminated state
    error, next_angle = vrep.simxGetFloatSignal(clientID, "angle",
                                                vrep.simx_opmode_buffer)
    error, next_velocity = vrep.simxGetFloatSignal(clientID, "velocity",
                                                   vrep.simx_opmode_buffer)
    error, reward = vrep.simxGetFloatSignal(clientID, "reward",
                                            vrep.simx_opmode_buffer)
    error, done = vrep.simxGetIntegerSignal(clientID, "done",
                                            vrep.simx_opmode_buffer)
    # integrate the angle and the velocity to the next obs
    observation_ = np.array([next_angle, next_velocity], dtype=np.float32)
    print(
        "lineNum{}|  angle:{:+.3f}  velocity:{:+.3f}  reward:{:+.3f}  done:{}".
        format(
            str(lineNum).zfill(3), next_angle * 180 / math.pi,
            next_velocity * 180 / math.pi, reward, done))
    return observation_, reward, done


def reset():
    # extract the first state from the vrep
    vrep.simxSetIntegerSignal(clientID, "line", 0, vrep.simx_opmode_blocking)
    error, angle = vrep.simxGetFloatSignal(clientID, "angle",
                                           vrep.simx_opmode_buffer)
    error, velocity = vrep.simxGetFloatSignal(clientID, "velocity",
                                              vrep.simx_opmode_buffer)
    obs = np.array([angle, velocity], dtype=np.float32)
    return obs


def stream():
    # prerequisite of retrieving the data from the buffer: declartion the mode in streaming
    error, next_angle = vrep.simxGetFloatSignal(clientID, "angle",
                                                vrep.simx_opmode_streaming)
    error, next_velocity = vrep.simxGetFloatSignal(clientID, "velocity",
                                                   vrep.simx_opmode_streaming)
    error, reward = vrep.simxGetFloatSignal(clientID, "reward",
                                            vrep.simx_opmode_streaming)
    error, done = vrep.simxGetIntegerSignal(clientID, "done",
                                            vrep.simx_opmode_streaming)


def visualize(EPISODE, Reward):
    plt.plot(EPISODE, Reward, c='r', label='DDPG')
    plt.legend(loc='best')
    plt.title('Reward')
    plt.ylabel('total reward')
    plt.xlabel('episode')
    plt.grid()
    plt.show()


####################  Main function  #########################

if __name__ == '__main__':
    # sate: angle and velocity
    s_dim = 2
    # action: dv/dt
    action = [[-2.5, 2.5]]
    a_dim = len(action)
    a_bound = max(action[0])
    EPISODE = []
    REWARD = []

    ddpg = DDPG(a_dim, s_dim, a_bound, outputGraph=True, saveModel=True)
    ounoise = OUNoise(action)

    LOAD = ddpg.load_Model(SAVEDIR, load=LOAD)
    stream()  # request the command
    if not LOAD:
        for episode in range(MAX_EPISODES):
            while True:
                # poll the useless signal (to receive a message from server)
                vrep.simxGetIntegerSignal(clientID, 'line',
                                          vrep.simx_opmode_blocking)

                # check server state (within the received message)
                e = vrep.simxGetInMessageInfo(
                    clientID, vrep.simx_headeroffset_server_state)

                # check bit0
                not_stopped = e[1] & 1

                if not not_stopped:
                    break
                else:
                    print('not_stopped')
                    vrep.simxStopSimulation(clientID,
                                            vrep.simx_opmode_blocking)

            # IMPORTANT
            # you should always call simxSynchronous()
            # before simxStartSimulation()
            vrep.simxSynchronous(clientID, True)
            # then start the simulation:
            e = vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)
            print('start', e)

            s = reset()
            ounoise.reset()
            clean_step()
            print('episode ', episode, ' started')
            print('memory state:', ddpg.pointer)
            lineNum = 1
            R = 0
            while True:
                a_ = ddpg.choose_action(s)
                # add randomness to action selection for exploration
                a = ounoise.get_action(a_, lineNum)
                # interact with the vrep
                s_, r, done = step(a, lineNum)
                R += r
                # restore the information in the buffer
                ddpg.store_transition(s, a, r, s_)
                #print("step{}  s:{} a:{:.3f} r:{} s_:{:.3f}".format(lineNum, s, a, r, s_))
                lineNum += 1
                if ddpg.pointer > BATCH_SIZE:
                    #
                    ddpg.learn()
                if done == 1:
                    # pointer is the index of buffer
                    if ddpg.pointer > BATCH_SIZE:
                        ddpg.save_Model(SAVEDIR, R, episode)
                    print('episode ', episode, ' finished')
                    print('total reward:', R, '\n')
                    REWARD.append(R)
                    EPISODE.append(episode)
                    # stop the simulation
                    vrep.simxStopSimulation(clientID,
                                            vrep.simx_opmode_blocking)
                    time.sleep(1.2)
                    break
                s = s_
        ddpg.close()
        visualize(EPISODE, REWARD)

    else:
        vrep.simxStartSimulation(clientID, vrep.simx_opmode_blocking)
        s = reset()
        ounoise.reset()
        lineNum = 1
        while True:
            a_ = ddpg.choose_action(s)
            # add randomness to action selection for exploration
            a = ounoise.get_action(a_, lineNum)
            s_, r, done = step(a, lineNum)
            lineNum += 1
            if done == 1:
                vrep.simxStopSimulation(clientID, vrep.simx_opmode_blocking)
                time.sleep(1.2)
                break
            s = s_
        ddpg.close()
    vrep.simxFinish(clientID)
