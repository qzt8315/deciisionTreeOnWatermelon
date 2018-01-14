import numpy as np

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
    print(data)