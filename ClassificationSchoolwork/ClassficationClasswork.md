导入包

    import scipy.io as sio  #读取.mat文件
    import numpy as np
    import os
    
    from sklearn.model_selection import train_test_split  #数据集划分
    from sklearn.decomposition import PCA  #PCA降维
    from sklearn.svm import SVC  #SVC模型
    from sklearn import preprocessing  #数据处理模块
数据集读取，并对每个样本进行类别标记

    data = np.array([])
    n = 0
    target = np.array([])
    for i in os.listdir('train/'):
        a,b = os.path.splitext(i)
        new = sio.loadmat('train/'+i)[a]
        for j in range(new.shape[0]):
            target = np.append(target,n)#为每个样本赋予标签
        data = np.append(data,new)
        n = n+1
    data = np.reshape(data,(-1,200))
    target = np.reshape(target,(-1,1))

数据归一化

    scaler = preprocessing.StandardScaler().fit(data)
    data = scaler.transform(data)

数据降维

    pca = PCA(n_components='mle',svd_solver = 'full',copy=False)
    train_x = pca.fit_transform(train_x)
    dev_x = pca.transform(dev_x)

使用网格搜索 寻找最优参数

        from sklearn.model_selection import GridSearchCV
        param_grid = {"gamma":[0.001,0.01,0.1,1,10,100],
                     "C":[0.001,0.01,0.1,1,10,100]}
        print("Parameters:{}".format(param_grid))
    
        grid_search = GridSearchCV(SVC(),param_grid,cv=5) #实例化一个GridSearchCV类
        grid_search.fit(train_x,train_y.ravel()) #训练，找到最优的参数，同时使用最优的参数实例化一个新的SVC estimator。
        print("Test set score:	 {:.2f}".format(grid_search.score(dev_x,dev_y)))
        print("Best parameters:{}".format(grid_search.best_params_))
        print("Best score on train set: 		{:.2f}".format(grid_search.best_score_))

模型训练

	clf = SVC(decision_function_shape='ovo',C = 100,gamma = 0.01)
	clf.fit(train_x,train_y.ravel())

结果预测 

	test_data = sio.loadmat('data_test_final.mat')['data_test_final']# 导入测试集
	test_data = scaler.transform(test_data)
	test_data = pca.transform(test_data)# PCA数据降维
	predict_y = clf.predict(test_data) # 使用训练模型预测


	
	for i in range(len(predict_y)):
	if(predict_y[i] == 0):
	    predict_y[i] = 2
	elif(predict_y[i] == 1):
	    predict_y[i] = 3
	elif(predict_y[i] == 2):
	    predict_y[i] = 5
	elif(predict_y[i] == 3):
	    predict_y[i] = 6
	elif(predict_y[i] == 4):
	    predict_y[i] = 8
	elif(predict_y[i] == 5):
	    predict_y[i] = 10
	elif(predict_y[i] == 6):
	    predict_y[i] = 11
	elif(predict_y[i] == 7):
	    predict_y[i] = 12
	elif(predict_y[i] == 8):
	    predict_y[i] = 14  # 修改类别标号

将结果保存至csv文件

	import pandas as pd
	id1 = np.arange(len(predict_y))
	predict_y = np.c_[id1,predict_y]
	columns = ['id','y']
	save_file = pd.DataFrame(columns=columns, data=predict_y)
	save_file.to_csv('test_y.csv', index=False, encoding="utf-8") #将预测结果保存至csv文件