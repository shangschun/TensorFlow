# -*- coding: utf-8 -*- 2
import tensorflow as tf

# 声明w1,w2两个变量，通过seed参数设置随机种子
# 保证每次运行得到的结果都是一样的
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

# 暂时将输入的特征定义为一个常量，x是一个1*2的矩阵
x = tf.constant([[0.7,0.9]])

# 前向传播算法
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)
# 建立会话
sess = tf.Session()
# 对w1,w2进行初始化
sess.run(w1.initializer)
sess.run(w2.initializer)
# global_variables_initializer方法可以实现对所有变量的初始化
# init_op = tf.global_variables_initializer()
# sess.run(init_op)
# 输出[[ 3.95757794]]
print(sess.run(y))
sess.close()