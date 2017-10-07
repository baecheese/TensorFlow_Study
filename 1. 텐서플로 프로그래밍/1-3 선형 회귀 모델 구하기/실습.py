import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# -1.0~1.0 사이의 균등분포를 가진 무작위 값으로 초기화

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")
# placeholder 설정 및 이름 지정

hypothesis = W * X + b
# W와의 곱과 b와의 합을 통해 X와 Y의 관계를 설명

cost = tf.reduce_mean(tf.square(hypothesis - Y))
# 모든 데이터에 대한 손실값의 평균

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1) # learning_rate : 학습률
train_op = optimizer.minimize(cost)

# session 블록
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(100):
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})
        print(step, cost_val, sess.run(W), sess.run(b))
        """
        0 4.18845 [ 1.33084369] [-0.51752776]
        1 0.0937537 [ 1.22906733] [-0.54635966]
        2 0.0427649 [ 1.23381495] [-0.52871466]
        3 0.0401776 [ 1.22707355] [-0.51649773]
        4 0.0382625 [ 1.22173738] [-0.50402761]
        5 0.036445 [ 1.21639359] [-0.49191704]
        6 0.0347138 [ 1.21119308] [-0.48009107]
        ....(이하 생략)
        손실값(cost)이 점점 작아진다.
        """

    print("\n=======test========")
    print("X: 5, Y:", sess.run(hypothesis, feed_dict={X: 5}))
    # X: 5, Y: [ 5.00688982]
    print("X: 2.5, Y:", sess.run(hypothesis, feed_dict={X: 2.5}))
    # X: 2.5, Y: [ 2.50057292]

