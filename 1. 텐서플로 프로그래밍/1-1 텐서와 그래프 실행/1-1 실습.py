import tensorflow as tf

hello = tf.constant('Hello, TensorFlow')
print(hello) # Tensor("Const_10:0", shape=(), dtype=string)

a = tf.constant(10)
b = tf.constant(32)
c = tf.add(a, b)
print(c) # Tensor("Add:0", shape=(), dtype=int32)


sess = tf.Session()
print(sess.run(hello))
print(sess.run([a, b, c]))

sess.close()

"""
b'Hello, TensorFlow'
[10, 32, 42]
"""

