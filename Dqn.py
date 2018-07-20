import numpy as np
import tensorflow as tf
import random
import os
import Stock_State
from argparse import ArgumentParser


MAX_SCORE_QUEUE_SIZE = 10


def get_options():
    parser = ArgumentParser()
    parser.add_argument('--MAX_EPISODE', type=int, default=10001, help='max number of episodes iteration')
    parser.add_argument('--ACTION_DIM', type=int, default=3, help='number of actions one can take')
    parser.add_argument('--OBSERVATION_DIM', type=int, default=5, help='number of observations one can see')
    parser.add_argument('--GAMMA', type=float, default=0.9, help='discount factor of Q learning')
    parser.add_argument('--INIT_EPS', type=float, default=1.0, help='initial probability of randomly sampling action')
    parser.add_argument('--FINAL_EPS', type=float, default=1e-5, help='final probability for randomly sampling action')
    parser.add_argument('--EPS_DECAY', type=float, default=0.9995, help='epsilon decay rate') # 원래 0.95
    parser.add_argument('--EPS_ANNEAL_STEPS', type=int, default=10, help='steps interval to decay epsilon')
    parser.add_argument('--LR', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--MAX_EXPERIENCE', type=int, default=3999, help='size of experience replay memory')
    parser.add_argument('--BATCH_SIZE', type=int, default=256, help='mini batch size')
    parser.add_argument('--H1_SIZE', type=int, default=32, help='size of hidden layer 1') #원래 128
    parser.add_argument('--H2_SIZE', type=int, default=32, help='size of hidden layer 2')
    parser.add_argument('--H3_SIZE', type=int, default=32, help='size of hidden layer 3')
    options = parser.parse_args()
    return options



class QAgent:

    def __init__(self, options):
        self.W1 = self.weight_variable([options.OBSERVATION_DIM, options.H1_SIZE])
        self.b1 = self.bias_variable([options.H1_SIZE])
        self.W2 = self.weight_variable([options.H1_SIZE, options.H2_SIZE])
        self.b2 = self.bias_variable([options.H2_SIZE])
        self.W3 = self.weight_variable([options.H2_SIZE, options.H3_SIZE])
        self.b3 = self.bias_variable([options.H3_SIZE])
        self.W4 = self.weight_variable([options.H3_SIZE, options.ACTION_DIM])
        self.b4 = self.bias_variable([options.ACTION_DIM])

    # 이 부분은 수정 필요. 혹시 tensorflow에서 그냥 될지도?
    def xavier_initializer(self, shape):
        dim_sum = np.sum(shape)
        if len(shape) == 1:
            dim_sum += 1
        bound = np.sqrt(6.0 / dim_sum)
        return tf.random_uniform(shape, minval=-bound, maxval=bound)

    def weight_variable(self, shape):
        return tf.Variable(self.xavier_initializer(shape))

    def bias_variable(self, shape):
        return tf.Variable(self.xavier_initializer(shape))

    def add_value_net(self, options):
        observation = tf.placeholder(tf.float32, [None, options.OBSERVATION_DIM])
        h1 = tf.nn.relu(tf.matmul(observation, self.W1) + self.b1)
        h2 = tf.nn.relu(tf.matmul(h1, self.W2) + self.b2)
        h3 = tf.nn.relu(tf.matmul(h2, self.W3) + self.b3)
        Q = tf.squeeze(tf.matmul(h3, self.W4) + self.b4)
        return observation, Q

    def sample_action(self, Q, feed, eps, options):
        act_values = Q.eval(feed_dict = feed)
        if random.random() <= eps:
            action_index = random.randrange(options.ACTION_DIM)
        else:
            action_index = np.argmax(act_values)
        action = np.zeros(options.ACTION_DIM)
        action[action_index] = 1
        return action


model_name = "samsung2nh"

def train(env):

    options = get_options()
    agent = QAgent(options)
    sess = tf.InteractiveSession()

    obs, Q1 = agent.add_value_net(options)
    act = tf.placeholder(tf.float32, [None, options.ACTION_DIM])
    rwd = tf.placeholder(tf.float32, [None, ])
    next_obs, Q2 = agent.add_value_net(options)


    # loss function 설정하는 부분인데 time discount factor가 들어가있는지를 모르겠다 --> 적용된듯!!
    values1 = tf.reduce_sum(tf.multiply(Q1, act), reduction_indices=1)
    values2 = rwd + options.GAMMA * tf.reduce_max(Q2, reduction_indices=1)
    loss = tf.reduce_mean(tf.square(values1 - values2))
    train_step = tf.train.AdamOptimizer(options.LR).minimize(loss)

    init = tf.global_variables_initializer()
    sess.run(init)

    
    saver = tf.train.Saver()
    print(os.path.isfile(os.path.join(os.getcwd(), model_name+".index")))
    if os.path.isfile(os.path.join(os.getcwd(), model_name+".index")):
        saver.restore(sess, os.path.join(os.getcwd(), model_name))
        print("Model Restored")
        
        
    else:
        print("No model restored")

    print("Starting Training Model")



    feed = {}
    eps = options.INIT_EPS
    global_step = 0
    exp_pointer = 0
    
    learning_finished = False

    obs_queue = np.empty([options.MAX_EXPERIENCE, options.OBSERVATION_DIM])
    act_queue = np.empty([options.MAX_EXPERIENCE, options.ACTION_DIM])
    rwd_queue = np.empty([options.MAX_EXPERIENCE])
    next_obs_queue = np.empty([options.MAX_EXPERIENCE, options.OBSERVATION_DIM])

    score_queue = []

    for i_episode in range(options.MAX_EPISODE):

        state, observation = env.reset(0)
        done = False
        score = 0
        sum_loss_value = 0
        act_0_num = 0
        act_1_num = 0
        act_2_num = 0
        while not done:
            global_step += 1
            if global_step % options.EPS_ANNEAL_STEPS == 0 and eps > options.FINAL_EPS:
                eps = eps * options.EPS_DECAY
            
            obs_queue[exp_pointer] = observation
            action = agent.sample_action(Q1, {obs : np.reshape(observation, (1, -1))}, eps, options)
            if np.argmax(action) == 0:
                act_0_num += 1
            elif np.argmax(action) == 1:
                act_1_num += 1
            elif np.argmax(action) == 2:
                act_2_num += 1

            act_queue[exp_pointer] = action
            
            observation, state, reward, done = env.step(np.argmax(action))
            
            score += reward
            reward = score
            
            rwd_queue[exp_pointer] = reward
            
            next_obs_queue[exp_pointer] = observation
            
            exp_pointer += 1
            if exp_pointer == options.MAX_EXPERIENCE:
                exp_pointer = 0
            
            if global_step >= options.MAX_EXPERIENCE:
                
                rand_indexs = np.random.choice(options.MAX_EXPERIENCE, options.BATCH_SIZE)
                feed.update({obs : obs_queue[rand_indexs]})
                feed.update({act : act_queue[rand_indexs]})
                feed.update({rwd : rwd_queue[rand_indexs]})
                feed.update({next_obs : next_obs_queue[rand_indexs]})
                if not learning_finished:
                    step_loss_value, _ = sess.run([loss, train_step], feed_dict = feed)
                else:
                    step_loss_value = sess.run(loss, feed_dict = feed)
                sum_loss_value += step_loss_value
                

        print("=== Episode {} ended with score = {}, avg_loss = {} ===, action_num = {}, {}, {}".format(i_episode+1, score, sum_loss_value / score, act_0_num, act_1_num, act_2_num))
        score_queue.append(score)
        if len(score_queue) > MAX_SCORE_QUEUE_SIZE:
            score_queue.pop(0)
            if np.mean(score_queue) > 100000000000000000000000000:
                learning_finished = True
            else:
                learning_finished = False
        if learning_finished:
            print("Testing!!!")
        if i_episode>0 and i_episode % 1000 == 0:
            saver.save(sess, os.path.join(os.getcwd(), model_name), global_step = i_episode)

    
if __name__ == "__main__":
    env = Stock_State.Pair_StockSpace()
    train(env)


