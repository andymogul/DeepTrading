import numpy as np
import tensorflow as tf
import random
import os
import Stock_State
import Dqn

from argparse import ArgumentParser


model_name = ""

def Portfolio(env):

    options = Dqn.get_options()
    agent = Dqn.QAgent(options)
    sess = tf.InteractiveSession()

    obs, Q1 = agent.add_value_net(options)
    act = tf.placeholder(tf.float32, [None, options.ACTION_DIM])
    rwd = tf.placeholder(tf.float32, [None, ])
    next_obs, Q2= agent.add_value_net(options)

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
        
        state, observation = env.reset(1)
        done = False
        act_queue = np.empty([217])
        portfolio_queue = np.empty([217])
        stock_num = np.empty([217])

        date_pointer = 0
        while(not done):
            
            action = agent.sample_action(Q1, {obs : np.reshape(observation, (1, -1))}, 0.0, options)
            
            if np.argmax(action) == 0:
                act_queue[date_pointer] = 1
            elif np.argmax(action) == 1:
                act_queue[date_pointer] = -1
            else:
                act_queue[date_pointer] = 0


            observation, state, reward, done = env.step(np.argmax(action))
            stock_num[date_pointer] = state[2]
            portfolio_queue[date_pointer] = state[3]
            date_pointer += 1
        print(date_pointer)
        np.savetxt(model_name+"_act.csv", act_queue, delimiter=",")
        np.savetxt(model_name+"_portfolio.csv", portfolio_queue, delimiter=",")
        np.savetxt(model_name+"t_stock.csv", stock_num, delimiter=",")

        

    else:    
        print("There's no checkpoint file")

if __name__ == "__main__":
    env = Stock_State.Single_StockSpace()
    Portfolio(env)