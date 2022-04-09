"""
reference: https://github.com/borgwang/toys/blob/master/visualization-convolution/
"""

import numpy as np

def gen_stu(size=5, mode="uniform"):
    """
    生成初始学生矩阵
        params: size (input int) 矩阵大小
        params: mode (input array) 生成方式
        return: stu_matrix (np.array) 生成学生矩阵
    """
    if mode == "uniform":
        return np.random.uniform(0, 1, [size, size])
    elif mode == "normal":
        return np.random.normal(0, 1, [size, size])
    else: 
        raise NotImplementedError

def get_padding(ks, mode="SAME"):
    """
    确定padding大小 卷人神经网络的
        params: inputs (input array)
        params: ks (kernel size) [p, q]
        params: mode (string) 根据输出的矩阵大小是否变化来确定；SAME: 大小不变；VALID: 做任何padding；
        return: padding list [n,m,j,k] in different modes
    """
    pad = None
    if mode == "FULL":
        pad = [ks[0] - 1, ks[1] - 1, ks[0] - 1, ks[1] - 1]
    elif mode == "VALID":
        pad = [0, 0, 0, 0]
    elif mode == "SAME":
        pad = [(ks[0] - 1) // 2, (ks[1] - 1) // 2, 
               (ks[0] - 1) // 2, (ks[1] - 1) // 2]
        if ks[0] % 2 == 0:
            pad[2] += 1
        if ks[1] % 2 == 0:
            pad[3] += 1
    else: 
        print("Invalid mode")
    return pad

def normalize_func(m):
    """
    对输入进行归一化
        params: inputs (input array)
        return: output matrix (input array)
    """
    mean = np.mean(m)
    std = np.std(m)

    return (m-mean) / std

def softmax(a):
	c = np.max(a)
	exp_a = np.exp(a - c) # 溢出对策
	sum_exp_a = np.sum(exp_a)
	y = exp_a / sum_exp_a
	return y

def ReLu(x, max_value=0):
    assert isinstance(x, np.ndarray), "x is not np.ndarray"
    x[x<=max_value] = max_value
    return x

def get_similar_value_ind(window, max_value, y, x, pad, height, width):
    """
    获取窗口最大值在原来map中的index
    """
    tmp = np.abs(window - max_value)
    y_, x_  = np.where(tmp==np.min(tmp))
    y_, x_ = np.array([y_,x_]).reshape(-1).tolist()
    y = y_ + y - pad[0]
    x = x_ + x - pad[0]

    # check vaild 如果索引超出了原来输入的map的索引，那么说明padding是最大值则返回None
    if (y<0 or x<0) or (y>=height or x >= width):
        return None

    return y, x