'''
HelloWorld example using TensorFlow library.
'''

import tensorflow as tf

# create constant
hello = tf.constant('Hello, TensorFlow!')
# create session to run
sess = tf.Session()
print(sess.run(hello))
