# -*- coding: utf-8 -*- 2
import tensorflow as tf
g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable(
        name="v",shape=[1],initializer = tf.zeros_initializer())
g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable(
        "v",shape = [1],initializer = tf.ones_initializer()
    )
with tf.Session(graph = g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("",reuse=True):
        print(sess.run(tf.get_variable("v")))
with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("",reuse=True):
        print(v.name)
        print(sess.run(tf.get_variable("v")))
# g1 = tf.Graph()
# g1 = tf.get_variable("v",shape=[1],initializer=tf.zeros_initializer)
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     print(sess.run(g1))