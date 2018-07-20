import numpy as np
import tensorflow as tf
import Stock_State
import Dqn
import os

env = Stock_State.Single_StockSpace()
options = Dqn.get_options()
agent = Dqn.QAgent(options)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(os.getcwd(), 'single_stock_checkpoint'))
    print("Model restored")
    print(sess())