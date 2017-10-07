import tensorflow as tf

X = tf.placeholder(tf.float32, [None, 3])
print(X)
# Tensor("Placeholder:0", shape=(?, 3), dtype=float32)

X_data = [[1, 2, 3], [4, 5, 6]]
# X는 [2, 3] 행렬 형태

W = tf.Variable(tf.random_normal([3, 2]))
b = tf.Variable(tf.random_normal([2, 1]))
# W는 [3, 2], b는 [2, 1] 행렬 형태로 정규 분포의 무작위 값으로 초기화

expr = tf.matmul(X, W) + b
print(expr)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 앞서 정의한 변수들을 초기화 - 기존 학습 값이 아닌 처음 실행하는 것이라면 반드시 필요

print("=== x_data ===")
print(x_data)
print("=== W ===")
print(sess.run(W))
print("=== b ===")
print(sess.run(b))
print("=== expr ===")
print(sess.run(expr, feed_dict={X: x_data}))

sess.close()

"""
=== x_data ===
[[1, 2, 3], [4, 5, 6]]
=== W ===
[[ 1.18953037 -0.70143777]
 [-0.11922577  1.78058231]
 [-1.20287728  0.35435316]]
=== b ===
[[ 1.32688916]
 [ 0.00204834]]
=== expr ===
[[-1.3306638   5.24967527]
 [-3.05322289  8.22532749]]
"""

