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

    def __init__(self, size):
        """
            Class (replay) memory constructor
            - size is the maximum number of element to store
            - obs_dim is the dimension of the state space
            - act_dim is the dimension of the action space
        """
        self.size = size
        self.ptr = 0
        self.count = 0
        self.state = np.zeros([size, 84, 84, 4], dtype=np.float32)
        self.action = np.zeros([size], dtype=np.int32)
        self.reward = np.zeros([size], dtype=np.float32)
        self.nextState = np.zeros([size, 84, 84, 4], dtype=np.float32)
        self.terminal = np.zeros([size], dtype=np.float32)

    def store(self, state, action, reward, next_state, end):
        """
            Store a transition into the replay memory
        """
        self.state[self.ptr, :, :, :] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.nextState[self.ptr, :, :, :] = next_state
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
        return {"state" : self.state[rd_idx, :, :, :],
                    "action" : self.action[rd_idx],
                    "reward" : self.reward[rd_idx],
                    "nextState" : self.nextState[rd_idx, :, :, :],
                    "terminal" : self.terminal[rd_idx]}

class Dqn:

    def __init__(self, env_fn,
                save_path = "./model/model.ckpt",
                epochs = 10000,
                max_steps_per_epoch = 1000,
                memorySize = int(5e4),
                batchSize = 128,
                gamma = 0.99,
                noise_floor = 0.02,
                noise_end = 50000,
                start = 1000,
                jump = 4,
                update_freq = 4,
                sync_freq = 1000,
                display_freq = 1000,
                save_freq = 100):
        """
        Constructor of the Dqn class
        - env_fn is the environment function
        - epochs is the number of epochs
        - max_steps_per_epoch is the number of steps in an epoch
        - memorySize is the number max of transition to store in the replay buffer
        - batchSize is the size of the batch
        - gamma is the target discount
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
        self.obs_dim = self.env.observation_space.shape
        self.act_dim = self.env.action_space.n
        self.epochs = epochs
        self.max_steps_per_epoch = max_steps_per_epoch
        self.memorySize = memorySize
        self.batchSize = batchSize
        self.gamma = gamma
        self.noise_floor = noise_floor
        self.noise_end = noise_end
        self.start = start
        self.jump = jump
        self.update_freq = update_freq
        self.sync_freq = sync_freq
        self.display_freq = display_freq
        self.save_freq = save_freq
        self.stateQueue = deque()
        self.state_dim = [None, 84, 84, 4]
        self.memory = Memory(self.memorySize)
        self.kernels = [[8,8], [4,4], [3,3]]
        self.strides = [(4,4), (2,2), (2,2)]
        self.filters = [16, 32, 64]
        self.fc_units = 256

    def defNet(self, input, activation, activation_out):
        for nbfilters, ksizes, stride in zip(self.filters, self.kernels, self.strides):
            input = tf.layers.conv2d(input, filters = nbfilters, strides = stride, kernel_size = ksizes, activation=activation)
        input_flat = tf.reshape(input, [-1, 4 * 4 * 64])
        input_flat = tf.layers.dense(input_flat, self.fc_units, activation = activation)
        return tf.layers.dense(input_flat, self.act_dim, activation = activation_out)

    def addNoise(self, action, step):
        """
        add noise from random normal distibution to an action scale with factor 0.1
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

    def initSeq(self):
        """
        Stack the four first frames of an episode
        """
        self.stateQueue = deque()
        obs = self.env.reset()
        for _ in range(4):
            state = self.sess.run(self.resized, feed_dict= {self.obs_ph: obs})
            self.stateQueue.append(state)
            for _ in range(self.jump):
                obs, reward, end, _ = self.env.step(self.env.action_space.sample())

    def build_graph(self):
        #Placeholders
        self.obs_ph  = tf.placeholder(tf.float32, self.obs_dim)
        self.state_ph = tf.placeholder(tf.float32, self.state_dim)
        self.action_ph = tf.placeholder(tf.int32, [None])
        self.reward_ph = tf.placeholder(tf.float32, [None])
        self.next_state_ph = tf.placeholder(tf.float32, self.state_dim)
        self.terminal_ph = tf.placeholder(tf.bool, [None])
        #Nets
        with tf.variable_scope('Q') as scope:
            self.q = self.defNet(self.state_ph, tf.nn.relu, None)
        with tf.variable_scope('Target_Q') as scope:
            self.q_target = self.defNet(self.next_state_ph, tf.nn.relu, None)
        #Utils Ops
        grayscale = tf.image.rgb_to_grayscale(self.obs_ph)
        self.resized = tf.squeeze(tf.image.resize_images(grayscale, (84,84)))
        self.sync_vars = tf.group([tf.assign(var2, var) for var, var2 in zip(self.get_vars('Q'), self.get_vars('Target_Q'))])
        self.max_q_next = tf.squeeze(tf.reduce_max(self.q_target, axis=1))
        self.targets = tf.stop_gradient(tf.clip_by_value(self.reward_ph, -1, 1) + tf.where(self.terminal_ph, tf.zeros(tf.shape(self.max_q_next)), self.gamma*self.max_q_next))
        self.select_action = tf.squeeze(tf.batch_gather(self.q, tf.expand_dims(self.action_ph, axis=1)))
        self.Qloss = tf.reduce_mean((self.select_action - self.targets)**2)
        self.Qtrain = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.Qloss, var_list = self.get_vars('Q'))
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
        self.initSeq()
        reward = 0
        for _ in range(nb_steps):
            state = np.stack(self.stateQueue, axis=2)
            action = np.argmax(self.sess.run(self.q, feed_dict={self.state_ph : np.expand_dims(state, 0)}))
            for _ in range(self.jump):
                self.env.render()
                obs, r, end, _ = self.env.step(action)
                reward += r
            next_state = self.sess.run(self.resized, feed_dict= {self.obs_ph: obs})
            self.stateQueue.popleft()
            self.stateQueue.append(next_state)
            if end:
                self.env.render()
                print("Episode reward : {}".format(reward))
                self.initSeq()
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
            self.initSeq()
            ep_ret = 0
            for step in range(self.max_steps_per_epoch):
                reward = 0
                global_step += 1
                #collect trajectories
                state = np.stack(self.stateQueue, axis=2)
                feed_dict = {self.state_ph: np.expand_dims(state, 0)}
                action = self.sess.run([self.q], feed_dict = feed_dict)
                action = self.addNoise(action, global_step)
                for _ in range(self.jump):
                    next_obs, r, end, _ = self.env.step(action)
                    reward += r
                u_next_state = self.sess.run(self.resized, feed_dict= {self.obs_ph: next_obs})
                self.stateQueue.popleft()
                self.stateQueue.append(u_next_state)
                next_state = np.stack(self.stateQueue, axis=2)
                end = False if step==self.max_steps_per_epoch-1 else end
                self.memory.store(state, action, reward, next_state, end)
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
            if epoch % self.save_freq == 0:
                save_path = self.saver.save(self.sess, self.save_path)
                print("Model saved in path: %s" % save_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pong-v0')
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
