import tensorflow as tf
import numpy as np

# [털, 날개] -> 없으면 0, 있으면 1
x_data = np.array (
    [[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]]
)

y_data = np.array ([
    [1, 0, 0], # 기타
    [0, 1, 0], # 포유류
    [0, 0, 1], # 조류
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
])

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 가중치(W) : [입력층(특징 수), 출력층(레이블 수)] 구성
# 편향 변수(b) : 레이블 수
W = tf.Variable(tf.random_uniform([2, 3], -1., 1.))
b = tf.Variable(tf.zeros([3]))

L = tf.add(tf.matmul(X, W), b)
L = tf.nn.relu(L) # 활성화 함수 ReLU 적용

# softmax : 배열 내의 결괏값들이 전체 합이 1이 되도록 만들어줌 *확률로 해석할수 있음
model = tf.nn.softmax(L)

# 교차 엔트로피로 손실값 구하기
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))

## 학습 시작
# 경사 하강법으로 최적화
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

# 텐서플로의 세션을 초기화
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 학습 100번 진행
for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})
    
    # 학습 도중 10번에 한 번씩 손실값 측정
    if (step + 1) % 10 == 0 :
        print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))
        

# 학습 결과 확인
# argmax : 요소 중 가장 큰 값의 인덱스를 찾아주는 함수
prediction = tf.argmax(model, axis=1)
target = tf.argmax(Y, axis=1)
print('예측값 : ', sess.run(prediction, feed_dict={X: x_data}))
print('실제값 : ', sess.run(target, feed_dict={Y: y_data}))

# 정확도 측정
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도 : %2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}) )


"""
10 1.17751
20 1.16955
30 1.16178
40 1.15418
50 1.14675
60 1.13948
70 1.13236
80 1.12687
90 1.12217
100 1.11834
예측값 :  [0 1 1 0 0 0]
실제값 :  [0 1 2 0 0 2]
정확도 : 66.666672
"""

