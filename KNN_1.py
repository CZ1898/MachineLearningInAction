# 所需模块
from numpy import *
import operator # 运算符模块

# 创建数据集和标签
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    # numpy中的array函数，直接创建多维数组
    # 注意array的括号,括号表明了这个是数组还是元组
    # 和明显可以看出包括两组，近1和近0
    labels = ['A','A','B','B']
    return group,labels

# k-nearest-neighbor
def classify0(inX,dataSet,labels,k):
    # inX:输入向量，训练样本集dataSet,标签向量labels,k选择邻近数目
    dataSetSize = dataSet.shape[0]

    # shape函数用法，a.shape 可以得到（xL,yL）(数组维度)(数组维数，数组里面列表维数)
    # a.shape[0],只输出行数，a.shape[1],只输出列数，元组的输出为(维数，)，和数组不同

    diffMat = tile(inX,(dataSetSize,1)) - dataSet   # 得到一个矩阵，矩阵为二者坐标之差

    # tile函数，tile就是瓷砖的意思
    # tile(mat,(a,b))重复mat矩阵a行b列组装
    # 这里就是将输入的数组变成和datasetsize一样

    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)  # 得到一个距离的array([d1,d2...])
    # 在numpy函数中 np.sum(mat,axis=0/1),
    # 0表示按行相加,不保持二维特性，若要保持二维特性，就用keepdims=True

    distances = sqDistances ** 0.5

    sortedDisIndicies = distances.argsort()
    # 将数组从小到大排列(索引值)

    # 建立一个空字典，K个邻近数值，之前已经从小到大排列，直接找到索引值
    # 显然索引值就代表了位置，且和label是一一对应的，故可直接获取此时的标签
    # 用get函数，get原本是访问字典项的访问num0flabel项，如果没有返回0
    # 首先进行了分类，然后分类之后还进行相加
    # 当开始空字典，一方面建立了classCount的类别，另一方面还得到了数字
    # 后面找得到，直接加1，可以说这段代码非常巧妙
    
    classCount = {}
    for i in range(k):
        num0flabel = labels[sortedDisIndicies[i]]
        classCount[num0flabel] = classCount.get(num0flabel,0) +1
        # 这句代码相当经典
    
    # 字典变数组，数组再排序
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    # sorted函数默认为升序，reverse就是降序了
    # iteritems 和 items 的作用大致相同，只是items返回列表，而iteritems返回迭代器
    # 通过operater的itemgetter函数得到，以下标为1的进行排序，1是索引值

    return sortedClassCount[0][0]

group,labels = createDataSet()
c = classify0([0.1,0.5],group,labels,4)
print(c)