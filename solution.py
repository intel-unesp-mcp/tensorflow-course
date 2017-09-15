"""
    Solution
"""

import os
import numpy as np
import tensorflow as tf

data = tf.placeholder(tf.int32, shape=(90, 2), name='data')
score = tf.reduce_sum(data, axis=0, name='score')

tie = tf.equal(score[0], score[1], name='tie')
win = tf.greater(score[0], score[1], name='win')


with tf.Session() as sess:
    writer = tf.summary.FileWriter("graphs", sess.graph)

    points = tf.Variable([0, 0], name='points')

    sess.run(tf.global_variables_initializer())

    for i in range(1, 100):

        X = np.loadtxt(os.path.join('data', 'jogo%d.txt') % i)

        print("Score game %d:" % i, score.eval(feed_dict={data: X}))

        if tie.eval(feed_dict={data: X}):
            sess.run(tf.assign_add(points, [1, 1]))

        elif win.eval(feed_dict={data: X}):
            sess.run(tf.assign_add(points, [3, 0]))

        else:
            sess.run(tf.assign_add(points, [0, 3]))

    print("Final points:", points.eval())


writer.close()
