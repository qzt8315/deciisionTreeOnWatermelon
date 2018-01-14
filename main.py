import numpy as np
import math

# 信息熵
def Ent(D = None, p = None):
    """
    :param D: 一维list
    :param p: 计算概率
    :return: 返回信息熵
    """
    ret = 0.
    if D is not None:
        count = len(D) * 1.
        p1 = {}
        for i in D:
            if i in p1.keys():
                p1[i] += 1
            else:
                p1[i] = 1
        for i in p1.keys():
            p1[i] /= count
            ret -= p1[i] * math.log2(p1[i])
        # 显示计算出来概率
        print(p1)
    elif p is not None:
        ret -= p * math.log2(p)
        ret -= (1 - p) * math.log2(1 - p)
    # 显示计算出来的ret
    # print(ret)
    return ret

def Gain(D, a):
    """
    :param D: 标签列表，一维行向量
    :param a: 属性标签列表，一维行向量
    :return:返回信息增益
    """
    ret = Ent(D)

    return ret

# 程序开始
if __name__ == "__main__":
    # 从文本文件中读取数据
    f = open('data/data.txt')
    data = []
    # 按行读取数据
    line = f.readline()
    while line:
        data = np.append(data,[float(i) for i in line.split()])
        line = f.readline()
    # 调整矩阵的维数
    data = data.reshape((14,6))
    # 测试调整的结果
    # print(data)
    # 获得数据集和标签集
    dataset = data[:, 1:5].tolist()
    labelset = data[:, 5].tolist()
    # 显示测试结果
    # print(dataset, '\n', labelset)
    # 测试信息熵的计算函数
    print(Ent(labelset))