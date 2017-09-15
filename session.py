"""
    Get value of a tensor in a tf.Session
"""

import tensorflow as tf


a = tf.constant(3, name='a')
b = tf.constant(4, name='b')
c = tf.add(a, b, name='c')
d = tf.multiply(a, b, name='d')

with tf.Session() as sess:
    print("\nEvaluate tensor a:", a.eval())
    print("\nEvaluate tensor b:", b.eval())
    print("\nEvaluate tensors a, b, c, d, simultaneusly:")
    a_val, b_val, c_val, d_val = sess.run([a, b, c, d])
    print("\n%d + %d = %d" % (a_val, b_val, c_val))
    print("\n%d * %d = %d" % (a_val, b_val, d_val))
