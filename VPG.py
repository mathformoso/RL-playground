import numpy as np
import tensorflow as tf
import gym
from gym.spaces import Box, Discrete
from collections import deque
import time

class Buffer:
    """
        Buffer to store visited transitions
    """

    def __init__(self, epochs_max_length, obs_dim, act_dim, action_space):
        """
            Class Buffer constructor
            - epochs_max_length is the maximum number of element in the buffer
            - obs_dim is the dimension of the state space
            - act_dim is the dimension of the action space
            - action_space is the action space
        """
        self.states = np.zeros((epochs_max_length,) + obs_dim)
        if isinstance(action_space, Box):
            self.actions = np.zeros((epochs_max_length, act_dim))
        else:
            self.actions = np.zeros([epochs_max_length, 1])
        self.rewards = np.zeros([epochs_max_length])
        self.values = np.zeros([epochs_max_length])
        self.buf_adv = np.zeros([epochs_max_length])
        self.buf_ret = np.zeros([epochs_max_length])
        self.start_ptr = 0
        self.ptr = 0

    def store(self, state, action, reward, value):
        """
            Store a transition into the replay memory
        """
        self.states[self.ptr, :] = state
        self.actions[self.ptr, :] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.ptr += 1

    def computeAdv(self, last_value, gam, lam):
        """
        Compute advantages estimations and rewards to go values
        """
        buf_adv_tmp = self.rewards[self.start_ptr:self.ptr-2] + gam*self.values[self.start_ptr+1:self.ptr-1] - self.values[self.start_ptr:self.ptr-2]
        buf_adv_tmp = np.append(buf_adv_tmp, last_value)
        self.buf_adv[self.start_ptr:self.ptr-1] = self.discountCumSum(buf_adv_tmp, lam*gam)
        self.buf_ret[self.start_ptr:self.ptr-1] = self.discountCumSum(self.rewards[self.start_ptr:self.ptr-1], gam)
        self.start_ptr = self.ptr

    def getAdv(self):
        """
        Get the normalized advantages values
        """
        mean_adv = np.mean(self.buf_adv)
        std_adv = np.std(self.buf_adv)
        return (self.buf_adv - mean_adv)/std_adv, self.buf_ret

    def discountCumSum(self, input, discount):
        """
        Compute efficiently the discounted cummulative sum
        - of the input vector
        - using discount factor
        """
        tmp = 0
        output = np.zeros(input.shape)
        for i, el in enumerate(input[::-1]):
            tmp = el + discount * tmp
            output[i] = tmp
        return output[::-1]

class Vpg:

    def __init__(self, env_fn,
                save_path,
                epochs = 500,
                epochs_max_length = 4000,
                gam = 0.99,
                lam = 0.97,
                hidden_units = [64, 64],
                learning_rate = 0.001,
                save_freq = 20):
        """
        Constructor of the Vpg class
        - env_fn is the environment function
        - save_path is the path where the model is saved, by default "./model/model.ckpt"
        - epochs is the number of update loop
        - epochs_max_length is the number of iterations in the collect trajectories loop
        - gam is the reward discount factor
        - lam is the advantage discount factor
        - hidden_units is a list containing the number of units in each hidden layer
        - learning_rate is the learning rate of Adam optimizer
        - save_freq is the number of epoch beetween consecutive saves
        """
        self.sess = tf.Session()
        self.env = env_fn()
        self.save_path = save_path
        self.obs_dim = self.env.observation_space.shape
        if isinstance(self.env.action_space, Box):
            self.act_dim = self.env.action_space.shape[-1]
        else:
            self.act_dim = self.env.action_space.n
        self.epochs = epochs
        self.epochs_max_length = epochs_max_length
        self.gamma = gam
        self.lambda_ = lam
        self.hidden_units = hidden_units
        self.lr = learning_rate
        self.save_freq = save_freq
        self.buf = Buffer(self.epochs_max_length, self.obs_dim, self.act_dim, self.env.action_space)

    def defNet(self, input, units, activation, activation_out):
        for sizes in units[:-1]:
            input = tf.layers.dense(input, sizes, activation=activation)
        logits = tf.layers.dense(input, units[-1], activation = activation_out)
        return logits

    def gaussian_likelihood(self, x, mu, log_std):
        """
        Compute the likelihood of input vector x
        - x input vector
        - mu mean of the ditribution
        - log_std log of the distribution std
        """
        pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std) + 1e-8))**2 + 2*log_std + np.log(2*np.pi))
        return tf.reduce_sum(pre_sum, axis=1)

    def sample(self, action, logits):
        """
        sample an action from logits
        and compute the probability of an input action
        """
        if isinstance(self.env.action_space, Discrete):
            log_proba = tf.nn.log_softmax(logits)
            pi = tf.multinomial(log_proba, 1)
            logp_a = tf.reduce_sum(tf.one_hot(tf.squeeze(action), depth = self.act_dim)*log_proba, axis = 1)
        elif isinstance(self.env.action_space, Box):
            log_std = tf.get_variable(name = 'log_std', dtype = tf.float32, initializer = -0.5*np.ones(self.act_dim, dtype = np.float32))
            std = tf.exp(log_std)
            mu = logits
            pi = mu + tf.random_normal(tf.shape(mu)) * std
            logp_a = self.gaussian_likelihood(action, mu, log_std)
        return (tf.squeeze(pi), logp_a)

    def build_graph(self):
        self.state_ph = tf.placeholder(tf.float32, [None, self.obs_dim[-1]])
        if isinstance(self.env.action_space, Box):
            self.action_ph = tf.placeholder(tf.float32)
        else:
            self.action_ph = tf.placeholder(tf.int64)
        self.adv_ph = tf.placeholder(tf.float32)
        self.ret_ph = tf.placeholder(tf.float32)
        logits = self.defNet(self.state_ph, self.hidden_units+[self.act_dim], tf.tanh, None)
        self.logits = tf.squeeze(logits)
        self.v = tf.squeeze(self.defNet(self.state_ph, self.hidden_units+[1], tf.tanh, None))
        self.action, logp_a = self.sample(self.action_ph, logits)
        self.actor_loss = -tf.reduce_mean(self.adv_ph * logp_a)
        self.critic_loss = tf.reduce_mean((self.ret_ph - self.v)**2)
        opt = tf.train.AdamOptimizer(learning_rate = self.lr)
        self.train_actor = opt.minimize(self.actor_loss)
        self.train_critic = opt.minimize(self.critic_loss)
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def play(self, nb_steps = int(1e4)):
        """
        play the results of training
        load model from tha path specified in Vpg class constructor
        - nb_steps is the length of the play
        """
        self.build_graph()
        self.saver.restore(self.sess, self.save_path)
        print("Model restored.")
        obs = self.env.reset()
        reward = 0
        for _ in range(nb_steps):
            self.env.render()
            if isinstance(self.env.action_space, Discrete):
                action = self.sess.run(self.logits, feed_dict={self.state_ph : obs.reshape(1,-1)})
                action = np.argmax(action)
            elif isinstance(self.env.action_space, Box):
                action = self.sess.run(self.logits, feed_dict={self.state_ph : obs.reshape(1,-1)})
            obs, r, end, _ = self.env.step(action)
            reward += r
            if end:
                self.env.render()
                obs = self.env.reset()
                print("Episode reward : {}".format(reward))
                reward = 0

    def run(self):
        """
        run VPG
        """
        self.build_graph()
        start_time = time.time()
        self.sess.run(self.init)
        obs = self.env.reset()
        Ep_buf = deque()
        ep_ret = 0
        for epoch in range(self.epochs):
            #collect trajectories
            for step in range(self.epochs_max_length):
                feed_dict = {self.state_ph: obs.reshape(1,-1)}
                action, v = self.sess.run([self.action, self.v], feed_dict = feed_dict)
                next_obs, reward, end, _ = self.env.step(action)
                self.buf.store(obs, action, reward, v)
                obs = next_obs
                ep_ret += reward
                if end :
                    self.buf.computeAdv(reward, self.gamma, self.lambda_)
                    obs = self.env.reset()
                    if len(Ep_buf) > 10: Ep_buf.popleft()
                    Ep_buf.append(ep_ret)
                    ep_ret = 0
                if step == self.epochs_max_length-1:
                    last_value = self.sess.run(self.v, feed_dict = {self.state_ph: obs.reshape(1,-1)})
                    self.buf.computeAdv(last_value, self.gamma, self.lambda_)
            #train
            advantages, returns = self.buf.getAdv()
            feed_dict = {
                    self.state_ph: self.buf.states,
                    self.action_ph: self.buf.actions,
                    self.adv_ph: advantages,
                    self.ret_ph: returns}
            _, _, loss_pi, loss_v = self.sess.run([self.train_actor, self.train_critic, self.actor_loss, self.critic_loss], feed_dict = feed_dict)
            #reset buffer
            self.buf = Buffer(self.epochs_max_length, self.obs_dim, self.act_dim, self.env.action_space)
            #Display logs
            print("***")
            print("Epoch:", epoch)
            print("EpRet:", np.mean(Ep_buf))
            print("LossPi:", loss_pi)
            print("LossV", loss_v)
            print("Time:", time.time()-start_time)
            print("***")
            #save
            if epoch % self.save_freq == 0:
                save_path = self.saver.save(self.sess, self.save_path)
                print("Model saved in path: %s" % save_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Cartpole-v1')
    parser.add_argument('--train', type=str, default='False')
    parser.add_argument('--path', type=str, default='./model/model.ckpt')
    args = parser.parse_args()
    vpg = Vpg(lambda : gym.make(args.env), args.path)
    if args.train == 'True':
        vpg.run()
    else:
        try:
            vpg.play()
        except tf.errors.NotFoundError as error:
            print("Error: no model to load")
