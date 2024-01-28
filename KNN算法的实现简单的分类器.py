import operator
import numpy as np
import matplotlib.pyplot as plt

#下面开始数据集的构建
def ceatedataset():
    group = np.array([[1.0,2.0],[1.2,0.1],[0.1,1.4],[0.3,3.5],[1.1,1.0],[0.5,1.5]])
    labels = np.array(['A','B','B','B','A','B'])
    return group,labels
if __name__=='__main__':
    group,labels = ceatedataset()
    plt.scatter(group[labels=='A',0],group[labels=='A',1],color = 'r',marker='*')
    plt.scatter(group[labels=='B',0],group[labels=='B',1],color = 'g',marker='+')
    plt.show()

#构建自己的简单的分类器
def KNN_classify(k,dis,X_train,Y_train,X_test):
    # assert dis == 'E' or dis == 'M'   'dis must E or M ; E代表欧式距离，M代表曼哈顿距离'
    num_test = X_test.shape[0]   #表示测试数据集的个数
    labellist = []
    '''
    使用欧式距离公式作为距离度量
    '''
    if (dis == 'E'):
        for i in range(num_test):
            #实现欧式距离的计算公式
            distances = np.sqrt(np.sum(((X_train - np.tile(X_test[i],(X_train.shape[0],1))) ** 2) , axis=1))
            nearest_k = np.argsort(distances) #距离由小到大进行排序，并返回 index 值
            topk = nearest_k[:k] # 选取前 K 个距离的点
            classCount = {}
            for j in topk:       # 依次遍历这 K 个点
                classCount[Y_train[j]] = classCount.get(Y_train[j],0) + 1
            # operator.itemgetter(1) 表示取出对象维度为 1 的数据
            sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
            labellist.append(sortedClassCount[0][0])
        return np.array(labellist)
    '''
    使用曼哈顿距离公式作为距离度量
    '''
    if (dis == 'M'):
        for i in range(num_test):
            #实现曼哈顿距离的计算公式
            distances = np.sum(np.abs(X_train - np.tile(X_test[i],(X_train.shape[0],1))),axis=1)
            nearset_k = np.argsort(distances)  #距离由大到小进行排序，并返回 index 值
            topk = nearset_k[:k] # 选取前 K 个距离的点
            '''
            下面开始进行类别计数，并对类别号进行排序，输出类别数目最多的作为该点的类别
            '''
            classCount = {}  # 定义了一个字典
            for j in topk:       # 依次遍历这 K 个点
                # 注意此处的 get 函数，因为没有设置 Y_train[j] 这个键的值，因此默认输出为 0
                classCount[Y_train[j]] = classCount.get(Y_train[j],0) + 1  #类别计算器，用来计算类别
            sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)  #对类别计数器进行排序，选择类别最多的一个作为该测试集的类别
            labellist.append(sortedClassCount[0][0])
        return np.array(labellist)

if __name__ == '__main__':
    group , labels = ceatedataset()
    X_test_pred = KNN_classify(k = 3,dis = 'M',X_train = group,Y_train = labels, X_test = np.array([[1.0,2.1],[0.4,2.0]]))
    print(X_test_pred)   #打印输出 ['A'.'B'],和我们的判断是一样的

'''
学习心得，感觉有点类似于相似得输出得理念，感觉有点类似与即时学习，通过计算与样本的距离来判断与样本的相似程度，进而得出距离最近的点，即：相似度最高
本算法的思量一般用于即时学习中，用来筛选相似度最高的数据
'''
