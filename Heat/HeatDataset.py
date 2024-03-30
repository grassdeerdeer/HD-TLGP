# coding:UTF-8
# @Time: 2023/8/14 10:53
# @Author: Lulu Cao
# @File: HeatDataset.py
# @Software: PyCharm
import numpy as np


def heat_2D_exact_solution(k,x1, x2, t):
    """

    :param k:
    :param x1: np.ndarray
    :param x2: np.ndarray
    :param t: np.ndarray
    :return:
    """
    return 1 / (4 * np.pi * k * t) * np.exp(-1 * (x1 ** 2 + x2 ** 2) / (4 * k * t))


def heat_3D_exact_solution(k,x1, x2, x3, t):
    """

    :param k:
    :param x1: np.ndarray
    :param x2: np.ndarray
    :param x3: np.ndarray
    :param t: np.ndarray
    :return:
    """
    return 1 / ((4 * np.pi * k * t) ** (3 / 2)) * np.exp(-1 * (x1 ** 2 + x2 ** 2 + x3 ** 2) / (4 * k * t))


def gen_exact_solution(d,k,x_dim,t_dim):
    """Generates exact solution for the heat equation for the given values of x and t."""
    # Number of points in each dimension:


    # Bounds of 'x' and 't':
    x_min, t_min = (0, 0.1)
    x_max, t_max = (1, 1.0)

    # Create tensors:
    t = np.linspace(t_min, t_max, num=t_dim).reshape(t_dim, 1)
    x = np.linspace(x_min, x_max, num=x_dim).reshape(x_dim, 1)

    if d == 2:
        usol = np.zeros((x_dim,x_dim, t_dim)).reshape(x_dim,x_dim, t_dim)
        # Obtain the value of the exact solution for each generated point:
        for i in range(x_dim):
            for m in range(x_dim):
                for j in range(t_dim):
                    usol[i][m][j] = heat_2D_exact_solution(k,x[i], x[m], t[j])

        # Save solution:
        np.savez("heat_2d_k"+str(k), x1=x,x2=x, t=t, usol=usol)

    if d == 3:
        usol = np.zeros((x_dim,x_dim,x_dim, t_dim)).reshape(x_dim,x_dim,x_dim,  t_dim)
        # Obtain the value of the exact solution for each generated point:
        for i in range(x_dim):
            for j in range(x_dim):
                for m in range(x_dim):
                    for n in range(t_dim):
                        usol[i][j][m][n] = heat_3D_exact_solution(k, x[i], x[j], x[m], t[n])

        # Save solution:
        np.savez("heat_3d_k"+str(k), x1=x, x2=x, x3=x, t=t, usol=usol)



import random
import numpy as np
import math


def dataset2D(sample=10,k=0.4):
    # 创建空列表存储x1、x2和y的值
    x1_list = []
    x2_list = []
    t_list = []
    y_list = []

    # 循环n次，随机生成n个采样点
    for i in range(sample):
        # 随机生成x1和x2的值
        x1 = random.uniform(0, 1)
        x2 = random.uniform(0, 1)
        t = random.uniform(0.1, 1)
        # 计算y的值
        y = 1 / (4 * math.pi * k * t) * math.exp(-1 * (x1 ** 2 + x2 ** 2) / (4 * k * t))
        # 将x1、x2和y的值添加到列表中
        x1_list.append(x1)
        x2_list.append(x2)
        t_list.append(t)
        y_list.append(y)


    return [x1_list,x2_list,t_list],y_list



def dataset3D(sample=15,k=0.4):
    # 创建空列表存储x1、x2和y的值
    x1_list = []
    x2_list = []
    x3_list = []
    t_list = []
    y_list = []

    # 循环n次，随机生成n个采样点
    for i in range(sample):
        # 随机生成x1和x2的值
        x1 = random.uniform(0, 1)
        x2 = random.uniform(0, 1)
        x3 = random.uniform(0, 1)
        t = random.uniform(0.1, 1)
        # 计算y的值
        y = 1 / (math.sqrt(math.pi*4*k)**3) * math.exp(-1 * (x1 ** 2 + x2 ** 2 + x3 ** 2) / (4 * k * t))
        # 将x1、x2和y的值添加到列表中
        x1_list.append(x1)
        x2_list.append(x2)
        x3_list.append(x3)
        t_list.append(t)
        y_list.append(y)


    return [x1_list,x2_list,x3_list,t_list],y_list
if __name__ == '__main__':
    d = 3
    k = 1
    x_dim = 20
    t_dim = 20
    gen_exact_solution(d, k, x_dim, t_dim)

