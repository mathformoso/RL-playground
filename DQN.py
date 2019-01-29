import numpy as np
import tensorflow as tf
import gym
from gym.spaces import Box, Discrete
from collections import deque
import time

class Memory:
    """
        Buffer to store visited transitions
    """

    def __init__(self, size, obs_dim, act_dim):
        """
            Class (replay) memory constructor
            - size is the maximum number of element to store
            - obs_dim is the dimension of the state space
            - act_dim is the dimension of the action space
        """
        self.size = size
        self.ptr = 0
        self.count = 0
        self.state = np.zeros([size, obs_dim], dtype=np.float32)
        self.action = np.zeros([size], dtype=np.int32)
        self.reward = np.zeros([size], dtype=np.float32)
        self.nextState = np.zeros([size, obs_dim], dtype=np.float32)
        self.terminal = np.zeros([size], dtype=np.float32)

    def store(self, state, action, reward, next_state, end):
        """
            Store a transition into the replay memory
        """
        self.state[self.ptr, :] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.nextState[self.ptr, :] = next_state
        self.terminal[self.ptr] = end
        self.ptr = (self.ptr + 1) % self.size
        self.count += 1

    def getRandomBatch(self, batchSize):
        """
            Select a set of random transitions in the replay memory
            - batchSize is the size of the set
        """
        #rd_idx = np.random.choice(min(self.size, self.count), batchSize, replace=False)
        rd_idx = np.random.randint(0, min(self.size, self.count), size=batchSize)
        return {"state" : self.state[rd_idx, :],
                    "action" : self.action[rd_idx],
                    "reward" : self.reward[rd_idx],
                    "nextState" : self.nextState[rd_idx, :],
                    "terminal" : self.terminal[rd_idx]}

class Dqn:

    def __init__(self, env_fn,
                save_path = "./model/model.ckpt",
                epochs = 10000,
                max_steps_per_epoch = 1000,
                memorySize = int(1e6),
                batchSize = 32,
                gamma = 0.99,
                units = [300],
                noise_floor = 0.02,
                noise_end = 50000,
                start = 1000,
                jump = 4,
                update_freq = 4,
                sync_freq = 1000,
                display_freq = 1000,
                save_freq = 200):
        """
        Constructor of the Dqn class
        - env_fn is the environment function
        - epochs is the number of epochs
        - max_steps_per_epoch is the number of steps in an epoch
        - memorySize is the number max of transition to store in the replay buffer
        - batchSize is the size of the batch
        - gamma is the target discount
        - units is a list containing the number of units in each hidden layer
        - noise floor : to select actions, noise is decreasing linearly from 1 to noise_floor after noise_end steps
        - noise_end : see noise_floor
        - start is the number of steps before to start training
        - jump is the number of skipped frames per step
        - update_freq is the training update frequency
        - sync_freq is the target synchronization frequency
        - display_freq is the infos display frequency
        - save_freq is the number of epoch beetween consecutive saves
        """
        self.env = env_fn()
        assert isinstance(self.env.action_space, Discrete)
        self.save_path = save_path
        self.sess = tf.Session()
        self.obs_dim = self.env.observation_space.shape[-1]
        self.act_dim = self.env.action_space.n
        self.epochs = epochs
        self.max_steps_per_epoch = max_steps_per_epoch
        self.memorySize = memorySize
        self.memory = Memory(self.memorySize, self.obs_dim, self.act_dim)
        self.batchSize = batchSize
        self.gamma = gamma
        self.units = units
        self.noise_floor = noise_floor
        self.noise_end = noise_end
        self.start = start
        self.jump = jump
        self.update_freq = update_freq
        self.sync_freq = sync_freq
        self.display_freq = display_freq
        self.save_freq = save_freq

    def defNet(self, input, units, activation, activation_out):
        for sizes in units[:-1]:
            input = tf.layers.dense(input, sizes, activation=activation)
        return tf.layers.dense(input, units[-1], activation = activation_out)

    def addNoise(self, action, step):
        """
        add epsilon noise to actions
        """
        eps = max(self.noise_floor, 1-step/self.noise_end)
        if np.random.rand() < eps:
            a = np.random.randint(self.act_dim)
        else:
            a = np.argmax(action)
        return a

    def get_vars(self, scope):
        """
        get vars from a scope
        """
        return [var for var in tf.global_variables() if scope in var.name]

    def build_graph(self):
        #Placeholders
        self.state_ph = tf.placeholder(tf.float32, [None, self.obs_dim])
        self.action_ph = tf.placeholder(tf.int32, [None])
        self.reward_ph = tf.placeholder(tf.float32, [None])
        self.next_state_ph = tf.placeholder(tf.float32, [None, self.obs_dim])
        self.terminal_ph = tf.placeholder(tf.bool, [None])
        #Nets
        with tf.variable_scope('Q') as scope:
            self.q = self.defNet(self.state_ph, self.units+[self.act_dim], tf.nn.relu, None)
        with tf.variable_scope('Target_Q') as scope:
            self.q_target = self.defNet(self.next_state_ph, self.units+[self.act_dim], tf.nn.relu, None)
        #Utils Ops
        self.sync_vars = tf.group([tf.assign(var2, var) for var, var2 in zip(self.get_vars('Q'), self.get_vars('Target_Q'))])
        self.max_q_next = tf.squeeze(tf.reduce_max(self.q_target, axis=1))
        self.targets = tf.stop_gradient(self.reward_ph + tf.where(self.terminal_ph, tf.zeros(tf.shape(self.max_q_next)), self.gamma*self.max_q_next))
        self.select_action = tf.squeeze(tf.batch_gather(self.q, tf.expand_dims(self.action_ph, axis=1)))
        self.Qloss = tf.reduce_mean((self.select_action - self.targets)**2)
        self.Qtrain = tf.train.AdamOptimizer().minimize(self.Qloss, var_list = self.get_vars('Q'))
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def play(self, nb_steps = int(1e4)):
        """
        play the results of training
        load model from tha path specified in Ddog class constructor
        - nb_steps is the length of the play
        """
        self.build_graph()
        self.saver.restore(self.sess, self.save_path)
        print("Model restored.")
        obs = self.env.reset()
        reward = 0
        for _ in range(nb_steps):
            self.env.render()
            action = np.argmax(self.sess.run(self.q, feed_dict={self.state_ph : obs.reshape(1,-1)}))
            obs, r, end, _ = self.env.step(action)
            reward += r
            if end:
                self.env.render()
                obs = self.env.reset()
                print("Episode reward : {}".format(reward))
                reward = 0

    def run(self):
        """
        run Q Learning
        """
        self.build_graph()
        start_time = time.time()
        self.sess.run(self.init)
        self.sess.run([self.sync_vars])
        global_step, cumuQloss = 0, 0
        Ep_buf = deque()
        #Main loop
        for epoch in range(self.epochs):
            ep_ret = 0
            obs = self.env.reset()
            for step in range(self.max_steps_per_epoch):
                global_step += 1
                #collect trajectories
                feed_dict = {self.state_ph: obs.reshape(1,-1)}
                action = self.sess.run(self.q, feed_dict = feed_dict)
                action = self.addNoise(action, global_step)
                for i in range(self.jump):
                    next_obs, reward, end, _ = self.env.step(action)
                end = False if step==self.max_steps_per_epoch-1 else end
                self.memory.store(obs, action, reward, next_obs, end)
                ep_ret += reward
                #update Q function
                if global_step % self.update_freq == 0 and global_step > self.start:
                    mem = self.memory.getRandomBatch(self.batchSize)
                    feed_dict = {self.state_ph : mem["state"],
                                self.reward_ph : mem["reward"],
                                self.action_ph : mem["action"],
                                self.next_state_ph : mem["nextState"],
                                self.terminal_ph : mem["terminal"]}
                    _, Qloss = self.sess.run([self.Qtrain, self.Qloss], feed_dict = feed_dict)
                    cumuQloss += Qloss
                if global_step % self.sync_freq == 0:
                    self.sess.run([self.sync_vars])
                if global_step % self.display_freq == 0:
                    #Display logs
                    print("***")
                    print("Epoch:", epoch)
                    print("Len:", global_step)
                    print("EpRet:", np.mean(Ep_buf))
                    print("LossQ:", cumuQloss/self.display_freq)
                    print("Time:", time.time()-start_time)
                    print("***")
                    cumuQloss = 0
                if end:
                    if len(Ep_buf) > 10:
                        Ep_buf.popleft()
                    Ep_buf.append(ep_ret)
                    break
                else: obs = next_obs
            if epoch % self.save_freq == 0:
                save_path = self.saver.save(self.sess, self.save_path)
                print("Model saved in path: %s" % save_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Acrobot-v1')
    parser.add_argument('--train', type=str, default='False')
    parser.add_argument('--path', type=str, default='./model/model.ckpt')
    args = parser.parse_args()
    try:
        dqn = Dqn(lambda : gym.make(args.env), args.path)
    except AssertionError as error:
        print("Error: action space is not discrete in this environment")
    if args.train == 'True':
        dqn.run()
    else:
        try:
            dqn.play()
        except tf.errors.NotFoundError as error:
            print("Error: no model to load")
