import numpy as np
import tensorflow as tf
import gym
from gym.spaces import Box, Discrete
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
        self.action = np.zeros([size, act_dim], dtype=np.float32)
        self.reward = np.zeros([size, 1], dtype=np.float32)
        self.nextState = np.zeros([size, obs_dim], dtype=np.float32)
        self.terminal = np.zeros([size, 1], dtype=np.float32)

    def store(self, state, action, reward, next_state, end):
        """
            Store a transition into the replay memory
        """
        self.state[self.ptr, :] = state
        self.action[self.ptr, :] = action
        self.reward[self.ptr, :] = reward
        self.nextState[self.ptr, :] = next_state
        self.terminal[self.ptr, :] = end
        self.ptr = (self.ptr + 1) % self.size
        self.count += 1

    def getRandomBatch(self, batchSize):
        """
            Select a set of random transitions in the replay memory
            - batchSize is the size of the set
        """
        rd_idx = np.random.randint(0, min(self.size, self.count), size=batchSize)
        return {"state" : self.state[rd_idx, :],
                    "action" : self.action[rd_idx, :],
                    "reward" : self.reward[rd_idx, :],
                    "nextState" : self.nextState[rd_idx, :],
                    "terminal" : self.terminal[rd_idx, :]}

class Ddpg:

    def __init__(self, env_fn,
                save_path,
                epochs = 300,
                steps_per_epochs = 1000,
                nb_updates = 1000,
                memorySize = int(1e6),
                batchSize = 128,
                poliakCst = 0.995,
                gamma = 0.99,
                units = [300],
                noise_factor = 0.1,
                start = 10,
                save_freq = 20):
        """
        Constructor of the Ddpg class
        - env_fn is the environment function
        - save_path is the path where the model is saved, by default "./model/model.ckpt"
        - epochs is the number of update loop
        - steps_per_epochs is the number of iterations in the collect trajectories loop
        - nb updates is the number of iterations in the update loop
        - memorySize is the number max of transition to store in the replay buffer
        - batchSize is the size of the batch
        - poliakCst is the constant to update targets
        - gamma is the target discount
        - units is a list containing the number of units in each hidden layer
        - noise factor scale the noise applied to actions from the policy
        - start is the number of epoch before parametrized policy starts exploring, before is random
        - save_freq is the number of epoch beetween consecutive saves
        """
        self.env = env_fn()
        assert isinstance(self.env.action_space, Box)
        self.save_path = save_path
        self.sess = tf.Session()
        self.obs_dim = self.env.observation_space.shape[-1]
        self.act_dim = self.env.action_space.shape[-1]
        self.act_limit = self.env.action_space.high[0]
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epochs
        self.nb_updates = nb_updates
        self.memorySize = memorySize
        self.memory = Memory(self.memorySize, self.obs_dim, self.act_dim)
        self.batchSize = batchSize
        self.k = poliakCst #poliak constant
        self.gamma = gamma
        self.units = units
        self.noise_factor = noise_factor
        self.start = start
        self.save_freq = save_freq

    def defNet(self, input, units, activation, activation_out):
        """
        net definition
        """
        for sizes in units[:-1]:
            input = tf.layers.dense(input, sizes, activation=activation)
        return tf.layers.dense(input, units[-1], activation = activation_out)

    def addNoise(self, action):
        """
        add noise from random normal distibution to an action
        """
        noise_a = action + self.noise_factor * np.random.randn(self.act_dim)
        return np.clip(noise_a, -self.act_limit, self.act_limit)

    def get_vars(self, scope):
        """
        get trainable vars from a scope
        """
        return [var for var in tf.trainable_variables() if scope in var.name]

    def build_graph(self):
        #Placeholders
        self.state_ph = tf.placeholder(tf.float32, [None, self.obs_dim])
        self.action_ph = tf.placeholder(tf.float32, [None, self.act_dim])
        self.reward_ph = tf.placeholder(tf.float32)
        self.terminal_ph = tf.placeholder(tf.bool)
        self.target_ph = tf.placeholder(tf.float32, [None, 1])
        #Nets
        with tf.variable_scope('Policy') as scope:
            self.pi = self.act_limit*self.defNet(self.state_ph, self.units+[self.act_dim], tf.nn.relu, tf.tanh)
        with tf.variable_scope('Target_Policy') as scope:
            self.pi_target = self.act_limit*self.defNet(self.state_ph, self.units+[self.act_dim], tf.nn.relu, tf.tanh)
        with tf.variable_scope('Q') as scope:
            self.q = self.defNet(tf.concat([self.state_ph, self.action_ph],1), self.units+[1], tf.nn.relu, None)
        with tf.variable_scope('Q', reuse=True) as scope:
            self.q_from_pi = self.defNet(tf.concat([self.state_ph, self.pi],1), self.units+[1], tf.nn.relu, None)
        with tf.variable_scope('Target_Q') as scope:
            self.q_target = self.defNet(tf.concat([self.state_ph, self.action_ph],1), self.units+[1], tf.nn.relu, None)
        with tf.variable_scope('Target_Q', reuse=True) as scope:
            self.q_from_pi_targ = self.defNet(tf.concat([self.state_ph, self.pi_target],1), self.units+[1], tf.nn.relu, None)
        #Utils Ops
        self.sync_vars1 = tf.group([tf.assign(var2, var) for var, var2 in zip(self.get_vars('Q'), self.get_vars('Target_Q'))])
        self.sync_vars2 = tf.group([tf.assign(var2, var) for var, var2 in zip(self.get_vars('Policy'), self.get_vars('Target_Policy'))])
        self.targets = self.reward_ph + tf.where(self.terminal_ph, tf.zeros(tf.shape(self.q_from_pi_targ)), self.gamma*self.q_from_pi_targ)
        self.Qloss = tf.reduce_mean((self.q - self.target_ph)**2)
        self.Qtrain = tf.train.AdamOptimizer().minimize(self.Qloss, var_list = self.get_vars('Q'))
        self.policyloss = -tf.reduce_mean(self.q_from_pi)
        self.policytrain = tf.train.AdamOptimizer().minimize(self.policyloss, var_list = self.get_vars('Policy'))
        self.updateQ = tf.group([tf.assign(var2, self.k*var2 + (1-self.k)*var) for var, var2 in zip(self.get_vars('Q'), self.get_vars('Target_Q'))])
        self.updatePolicy = tf.group([tf.assign(var2, self.k*var2 + (1-self.k)*var) for var, var2 in zip(self.get_vars('Policy'), self.get_vars('Target_Policy'))])
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
            action = self.sess.run(self.pi, feed_dict={self.state_ph : obs.reshape(1,-1)})
            obs, r, end, _ = self.env.step(action[0])
            reward += r
            if end:
                self.env.render()
                obs = self.env.reset()
                print("Episode reward : {}".format(reward))
                reward = 0

    def run(self):
        """
        run DDPG
        """
        self.build_graph()
        start_time = time.time()
        self.sess.run(self.init)
        self.sess.run([self.sync_vars1, self.sync_vars2])
        #Main loop
        for epoch in range(self.epochs):
            ep_steps, ep_ret, cumuQloss, cumuPiloss = 0, 0, 0, 0
            obs = self.env.reset()
            #collect trajectories
            for step in range(self.steps_per_epoch):
                if epoch > self.start:
                    feed_dict = {self.state_ph: obs.reshape(1,-1)}
                    action = self.sess.run(self.pi, feed_dict = feed_dict)
                    action = self.addNoise(action[0])
                else:
                    action = self.env.action_space.sample()
                next_obs, reward, end, _ = self.env.step(action)
                end = False if step==self.steps_per_epoch-1 else end
                self.memory.store(obs, action, reward, next_obs, end)
                ep_steps += 1
                ep_ret += reward
                if end: break
                else: obs = next_obs
            #update
            for update in range(self.nb_updates):
                #compute targets
                mem = self.memory.getRandomBatch(self.batchSize)
                feed_dict = {self.state_ph : mem["nextState"],
                            self.reward_ph: mem["reward"],
                            self.terminal_ph : mem["terminal"]}
                targets = self.sess.run([self.targets], feed_dict = feed_dict)
                #update Q function
                feed_dict = {self.state_ph : mem["state"],
                            self.action_ph : mem["action"],
                            self.target_ph: targets[0]}
                _, Qloss = self.sess.run([self.Qtrain, self.Qloss], feed_dict = feed_dict)
                cumuQloss += Qloss
                #update policy
                feed_dict = {self.state_ph : mem["state"]}
                _, piLoss = self.sess.run([self.policytrain, self.policyloss], feed_dict = feed_dict)
                cumuPiloss += piLoss
                #update target nets
                self.sess.run([self.updateQ, self.updatePolicy])
            #Display logs
            print("***")
            print("Epoch:", epoch)
            print("EpLen:", ep_steps)
            print("EpRet:", ep_ret)
            print("LossQ:", cumuQloss/self.nb_updates)
            print("LossPi:", cumuPiloss/self.nb_updates)
            print("Time:", time.time()-start_time)
            print("***")
            #save
            if epoch % self.save_freq == 0:
                save_path = self.saver.save(self.sess, self.save_path)
                print("Model saved in path: %s" % save_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--train', type=str, default='False')
    parser.add_argument('--path', type=str, default='./model/model.ckpt')
    args = parser.parse_args()
    try:
        ddpg = Ddpg(lambda : gym.make(args.env), args.path)
    except AssertionError as error:
        print("Error: action space is not continuous in this environment")
    if args.train == 'True':
        ddpg.run()
    else:
        try:
            ddpg.play()
        except tf.errors.NotFoundError as error:
            print("Error: no model to load")
