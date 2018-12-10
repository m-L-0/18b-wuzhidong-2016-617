导入包

```python
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
%matplotlib inline
```
加载数据集

	iris = load_iris()
	data = iris.data
	label = iris.target

数据集划分
	from sklearn.model_selection import train_test_split
	train_x, test_x, train_y, test_y = train_test_split(data,label,
                                                    	test_size=0.2,
                                                    	shuffle=True)

创建占位符，创建函数 通过寻找距离矩阵中的前K个近邻进行投票，决策出样本类别

```python
#设置占位符
xtr = tf.placeholder(tf.float32, shape=[None, 4])
xte = tf.placeholder(tf.float32, shape=[4])
#计算L1距离
dist = tf.sqrt(tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), 		 reduction_indices=1))
def knn(K): #K个近邻
    with tf.Session() as sess:
        predict = [] #存放所有测试样本的预测类别
        for i in range(len(test_x)):
            dist_mat = sess.run(dist, feed_dict={xtr:train_x, xte:test_x[i]})  
            # 将距离矩阵排序后，取出前Ｋ个近邻，进行投票决策
            knn_idx = np.argsort(dist_mat)[:K]       
            classes = [0, 0, 0]
            for idx in knn_idx:
                if(train_y[idx]==0):
                    classes[0] += 1
                elif(train_y[idx]==1):
                    classes[1] += 1
                else:
                    classes[2] += 1
            y_pred = np.argmax(classes)
            predict.append(y_pred)
        return predict
```
通过循环选取K值

	all_acc = [] #存放25个k值的正确率
	for K in range(25):
		y_pred = knn(K)
	   		 y_true = test_y
	    		acc = np.sum(np.equal(y_pred,y_true)) / len(y_true)
	   		 all_acc.append(acc)
	acc = max(all_acc) #求出20个 K值中正确率最高的
	print(acc) #输出正确率