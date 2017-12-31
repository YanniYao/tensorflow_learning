# 前向传播算法
import tensorflow as tf


# 声明w1、w2两个变量，这里还通过seed参数设定了随机种子，这样可以保证每次运行得到的结果是一样的
w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))


# 暂时将输入的特征变量定义为一个常量。
x = tf.constant([[0.7, 0.9]])


a = tf.matmul(x, w1)
y = tf.matmul(a, w2)


sess = tf.Session()


# 初始化w1,w2
sess.run(w1.initializer)
sess.run(w2.initializer)


print(sess.run(y))
sess.close()
