import os, time
import tensorflow.compat.v1 as tf
import numpy as np
import gym
from skimage import color, transform

# from tensorflow.initializers import random_uniformF
tf.disable_v2_behavior()

class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
                                                            self.mu, self.sigma)
from collections import deque
class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.memory = deque(maxlen=5000)
        self.mem_cntr = len(self.memory)
    def store_transition(self, state, action, reward, state_, done):
        self.memory.append([state, action, reward, new_state, done])

    def sample_buffer(self, batch_size):
        samples = random.sample(self.memory, batch_size)
        states, actions, rewards, states_, terminal = samples

        return states, actions, rewards, states_, terminal

class Actor(object):
    def __init__(self, lr, n_actions, name, input_dims, sess, fc1_dims,
                 fc2_dims, action_bound, batch_size=64):
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
       
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.sess = sess
        self.action_bound = action_bound
        self.build_network()
        self.params = tf.trainable_variables(scope=self.name)
        self.saver = tf.train.Saver()
        # self.checkpoint_file = os.path.join(chkpt_dir, name + '_ddpg.ckpt')

        self.unnormalized_actor_gradients = tf.gradients(
            self.mu, self.params, -self.action_gradient)

        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size),
                                        self.unnormalized_actor_gradients))

        self.optimize =  tf.train.AdamOptimizer(self.lr).\
                    apply_gradients(zip(self.actor_gradients, self.params))

    def build_network(self):
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32,
                                        shape=[None, *self.input_dims],
                                        name='inputs')

            self.action_gradient = tf.placeholder(tf.float32,
                                          shape=[None, self.n_actions],
                                          name='gradients')
            x = tf.keras.layers.Conv2D(16, kernel_size = 8, strides = (3,3), activation = 'relu')(self.input)
            conv1 = tf.keras.layers.Conv2D(32, kernel_size = 4, strides = (2,2), activation = 'relu')(x)
            flat = tf.layers.flatten(conv1)
            f1 = 0.05
            dense1 = tf.layers.dense(flat, units=self.fc1_dims,
                                     kernel_initializer=tf.keras.initializers.RandomUniform(minval = -f1, maxval = f1),
                                     bias_initializer=tf.keras.initializers.RandomUniform(minval = -f1, maxval = f1))
            batch1 = tf.layers.batch_normalization(dense1)
            layer1_activation = tf.nn.relu(batch1)
            f2 = 0.1
            dense2 = tf.layers.dense(layer1_activation, units=self.fc2_dims,
                                     kernel_initializer=tf.keras.initializers.RandomUniform(minval = -f2, maxval = f2),
                                     bias_initializer=tf.keras.initializers.RandomUniform(minval = -f2, maxval = f2))
            batch2 = tf.layers.batch_normalization(dense2)
            layer2_activation = tf.nn.relu(batch2)
            f3 = 0.003
            mu = tf.layers.dense(layer2_activation, units=self.n_actions,
                            activation='tanh',
                            kernel_initializer= tf.keras.initializers.RandomUniform(minval = -f3, maxval = f3),
                            bias_initializer=tf.keras.initializers.RandomUniform(minval = -f3, maxval = f3))
            mu2 = tf.layers.dense(layer2_activation, units=self.n_actions,
                            activation='sigmoid',
                            kernel_initializer= tf.keras.initializers.RandomUniform(minval = -f3, maxval = f3),
                            bias_initializer=tf.keras.initializers.RandomUniform(minval = -f3, maxval = f3))
            mu3 = tf.layers.dense(layer2_activation, units=self.n_actions,
                            activation='sigmoid',
                            kernel_initializer= tf.keras.initializers.RandomUniform(minval = -f3, maxval = f3),
                            bias_initializer=tf.keras.initializers.RandomUniform(minval = -f3, maxval = f3))
            output = tf.concat([mu, mu2, mu3], axis = 0)
            self.mu = tf.multiply(output, self.action_bound)

    def predict(self, inputs):
        return self.sess.run(self.mu, feed_dict={self.input: inputs})

    def train(self, inputs, gradients):
        self.sess.run(self.optimize,
                      feed_dict={self.input: inputs,
                                 self.action_gradient: gradients})

    def load_checkpoint(self):
        print("...Loading checkpoint...")
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print("...Saving checkpoint...")
        self.saver.save(self.sess, self.checkpoint_file)

class Critic(object):
    def __init__(self, lr, n_actions, name, input_dims, sess, fc1_dims, fc2_dims,
                 batch_size=64):
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        # self.chkpt_dir = chkpt_dir
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.sess = sess
        self.build_network()
        self.params = tf.trainable_variables(scope=self.name)
        self.saver = tf.train.Saver()
        # self.checkpoint_file = os.path.join(chkpt_dir, name +'_ddpg.ckpt')

        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.action_gradients = tf.gradients(self.q, self.actions)

    def build_network(self):
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32,
                                        shape=[None, *self.input_dims],
                                        name='inputs')

            self.actions = tf.placeholder(tf.float32,
                                          shape=[None, self.n_actions],
                                          name='actions')

            self.q_target = tf.placeholder(tf.float32,
                                           shape=[None,1],
                                           name='targets')
            x = tf.keras.layers.Conv2D(16, kernel_size = 8, strides = (3,3), activation = 'relu')(self.input)
            conv1 = tf.keras.layers.Conv2D(32, kernel_size = 4, strides = (2,2), activation = 'relu')(x)
            flat = tf.layers.flatten(conv1)

            f1 = 0.05
            dense1 = tf.layers.dense(flat, units=self.fc1_dims,
                                     kernel_initializer=tf.keras.initializers.RandomUniform(minval = -f1, maxval = f1),
                                     bias_initializer=tf.keras.initializers.RandomUniform(minval = -f1, maxval = f1))
            batch1 = tf.layers.batch_normalization(dense1)
            layer1_activation = tf.nn.relu(batch1)

            f2 = 0.1
            dense2 = tf.layers.dense(layer1_activation, units=self.fc2_dims,
                                     kernel_initializer=tf.keras.initializers.RandomUniform(minval = -f2, maxval = f2),
                                     bias_initializer=tf.keras.initializers.RandomUniform(minval = -f2, maxval = f2))
            batch2 = tf.layers.batch_normalization(dense2)
            #layer2_activation = tf.nn.relu(batch2)
            #layer2_activation = tf.nn.relu(dense2)

            action_in = tf.layers.dense(self.actions, units=self.fc2_dims,
                                        activation='relu')
            #batch2 = tf.nn.relu(batch2)
            # no activation on action_in and relu activation on state_actions seems to
            # perform poorly.
            # relu activation on action_in and relu activation on state_actions
            # does reasonably well.
            # relu on batch2 and relu on action in performs poorly

            #state_actions = tf.concat([layer2_activation, action_in], axis=1)
            state_actions = tf.add(batch2, action_in)
            state_actions = tf.nn.relu(state_actions)
            f3 = 0.003
            self.q = tf.layers.dense(state_actions, units=1,
                               kernel_initializer=tf.keras.initializers.RandomUniform(minval = -f3, maxval = f3),
                               bias_initializer=tf.keras.initializers.RandomUniform(minval = -f3, maxval = f3),
                               kernel_regularizer=tf.keras.regularizers.l2(0.01))

            self.loss = tf.losses.mean_squared_error(self.q_target, self.q)

    def predict(self, inputs, actions):
        return self.sess.run(self.q,
                             feed_dict={self.input: inputs,
                                        self.actions: actions})
    def train(self, inputs, actions, q_target):
        return self.sess.run(self.optimize,
                      feed_dict={self.input: inputs,
                                 self.actions: actions,
                                 self.q_target: q_target})

    def get_action_gradients(self, inputs, actions):
        return self.sess.run(self.action_gradients,
                             feed_dict={self.input: inputs,
                                        self.actions: actions})
    def load_checkpoint(self):
        print("...Loading checkpoint...")
        self.saver.restore(self.sess, self.checkpoint_file)

    def save_checkpoint(self):
        print("...Saving checkpoint...")
        self.saver.save(self.sess, self.checkpoint_file)

def rgb2gray(rgb):
    i = rgb[:84, 5:89, :]
    i = 2 * color.rgb2gray(i) - 1
    return i.reshape((84, 84, 1))

class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99, n_actions=3,
                 max_size=5000, layer1_size=400, layer2_size=300,
                 batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.sess = tf.Session()
        self.actor = Actor(alpha, n_actions, 'Actor', input_dims, self.sess,
                           layer1_size, layer2_size, env.action_space.high)
        self.critic = Critic(beta, n_actions, 'Critic', input_dims,self.sess,
                             layer1_size, layer2_size)

        self.target_actor = Actor(alpha, n_actions, 'TargetActor',
                                  input_dims, self.sess, layer1_size,
                                  layer2_size, env.action_space.high)
        self.target_critic = Critic(beta, n_actions, 'TargetCritic', input_dims,
                                    self.sess, layer1_size, layer2_size)

        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        # define ops here in __init__ otherwise time to execute the op
        # increases with each execution.
        self.update_critic = \
        [self.target_critic.params[i].assign(
                      tf.multiply(self.critic.params[i], self.tau) \
                    + tf.multiply(self.target_critic.params[i], 1. - self.tau))
         for i in range(len(self.target_critic.params))]

        self.update_actor = \
        [self.target_actor.params[i].assign(
                      tf.multiply(self.actor.params[i], self.tau) \
                    + tf.multiply(self.target_actor.params[i], 1. - self.tau))
         for i in range(len(self.target_actor.params))]

        self.sess.run(tf.global_variables_initializer())

        self.update_network_parameters(first=True)

    def update_network_parameters(self, first=False):
        if first:
            old_tau = self.tau
            self.tau = 1.0
            self.target_critic.sess.run(self.update_critic)
            self.target_actor.sess.run(self.update_actor)
            self.tau = old_tau
        else:
            self.target_critic.sess.run(self.update_critic)
            self.target_actor.sess.run(self.update_actor)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = state.reshape(-1, 84, 84, 1)
        mu = self.actor.predict(state) # returns list of list
        noise = self.noise()
        mu_prime = mu+ noise

        return mu_prime[0]

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = \
                                      self.memory.sample_buffer(self.batch_size)

        critic_value_ = self.target_critic.predict(new_state.reshape(-1, 84, 84, 1),
                                           self.target_actor.predict(new_state.reshape(-1, 84, 84, 1)))
        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j]*done[j])
        target = np.reshape(target, (self.batch_size, 1))

        _ = self.critic.train(state.reshape(-1, 84, 84, 1), action, target)

        a_outs = self.actor.predict(state.reshape(-1, 84, 84, 1))
        grads = self.critic.get_action_gradients(state.reshape(-1, 84, 84, 1), a_outs)

        self.actor.train(state, grads[0])

        self.update_network_parameters()

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

from gym.envs.registration import registry, register, make, spec

register(
    id='CarRacing-v1', # CHANGED
    entry_point='gym.envs.box2d:CarRacing',
    max_episode_steps=1200, # CHANGED
    reward_threshold=900,
)

def rgb2gray(rgb):
    i = rgb[:84, 5:89, :]
    i = 2 * color.rgb2gray(i) - 1
    return i.reshape((84, 84, 1))

env = gym.make('CarRacing-v1')
agent = Agent(alpha=0.00005, beta=0.0005, input_dims=[84,84,1], tau=0.001, env=env,
              batch_size=32,  layer1_size=400, layer2_size=300, n_actions=3)
np.random.seed(0)
#agent.load_models()
#env = wrappers.Monitor(env, "tmp/walker2d",
#                            video_callable=lambda episode_id: True, force=True)
print(env.action_space.high)
score_history = []
for i in range(1500):
    obs = env.reset()
    done = False
    score = 0
    state = rgb2gray(obs).reshape(-1, 84, 84, 1)
    while not done:
        
        act = agent.choose_action(state)
        new_state, reward, done, info = env.step(act)
        new_state = rgb2gray(new_state).reshape(-1, 84, 84, 1)
        agent.remember(obs, act, reward, new_state.reshape(84, 84, 1), int(done))
        agent.learn()
        score += reward
        obs = new_state
        env.render()
    score_history.append(score)
    print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))
    # if i % 25 == 0:
    #     agent.save_models()
with open('rewards.txt', 'w') as filehandle:
            for listitem in score_history:
                filehandle.write('%s\n' % str(listitem))

# filename = 'WalkerTF-alpha00005-beta0005-400-300-original-5000games-testing.png'
plt.plot(score_history)
