# -*- coding: utf-8 -*- 2
import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2,3],stddev=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1))
# 定义placeholder作为存放数据的地方，
x = tf.placeholder(tf.float32,shape=(1,2),name="input")
x1 = tf.placeholder(tf.float32,shape=(3,2),name="input")
a = tf.matmul(x1,w1)
y = tf.matmul(a,w2)

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
# print(sess.run(y))
# print(sess.run(y,feed_dict={x:[[0.7,0.9]]}))
print(sess.run(y,feed_dict={x1:[[0.7,0.9],[0.1,0.4],[0.5,0.8]]}))