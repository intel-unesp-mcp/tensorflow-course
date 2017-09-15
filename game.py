"""
    Let's play football
"""

import numpy as np
import tensorflow as tf

"""
    Define the tensor game
"""
game = tf.Variable(tf.zeros((90, 2), dtype=tf.int32), name='game')

"""
    Placeholder for the input data
"""
data = tf.placeholder(tf.int32, shape=(90, 2), name='data')

"""
    Operation to mutate the Tensor game
"""
goals = tf.assign_add(game, data, name='goals')

"""
    Operation to compute the final score
"""
score = tf.reduce_sum(game, axis=0, name='score')

"""
    Start a session
"""
with tf.Session() as sess:
    """
        Prepare Tensorboar logdir
    """
    writer = tf.summary.FileWriter('./graphs', sess.graph)

    """
        Try to read the tensor game
    """
    try:
        print(game.eval())
    except:
        print("\nPlease initialize the tensor")

    """
        Initialize tensor game
    """
    sess.run(game.initializer)
    print("\nInitialized tensor game")
    print(game.eval())

    """
        Load data
    """
    X = np.loadtxt('data/jogo1.txt')

    """
        Run the operation goals
    """
    sess.run(goals, feed_dict={data: X})
    print("\nTensor game after feeding the data")
    print(game.eval())

    """
        Evaluate and print the score
    """
    print("\nFinal score")
    print(score.eval())

"""
    Close writer
"""
writer.close()
