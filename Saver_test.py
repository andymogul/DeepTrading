import tensorflow as tf
import os

v = tf.Variable(tf.constant(0))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(os.getcwd(), 'trained_variables2'), global_step = 100)

    