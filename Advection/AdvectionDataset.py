# coding:UTF-8
# @Time: 2023/8/14 15:13
# @Author: Lulu Cao
# @File: AdvectionDataset.py
# @Software: PyCharm

# coding:UTF-8
# @Time: 2023/8/14 10:53
# @Author: Lulu Cao
# @File: HeatDataset.py
# @Software: PyCharm
import numpy as np


def advection1_2D_exact_solution(x1, x2, t):
    """
    :param x1: np.ndarray
    :param x2: np.ndarray
    :param t: np.ndarray
    :return:
    """
    return x1+x2-2*t


def advection2_3D_exact_solution(x1, x2, x3, t):
    """
    :param x1: np.ndarray
    :param x2: np.ndarray
    :param x3: np.ndarray
    :param t: np.ndarray
    :return:
    """
    return x1+x2+x3-3*t

def advection3_2D_exact_solution(x1, x2, t):
    """
    :param x1: np.ndarray
    :param x2: np.ndarray
    :param t: np.ndarray
    :return:
    """
    return np.sin(x1-t)+np.sin(x2-t)


def advection4_3D_exact_solution(x1, x2, x3, t):
    """
    :param x1: np.ndarray
    :param x2: np.ndarray
    :param x3: np.ndarray
    :param t: np.ndarray
    :return:
    """
    return np.sin(x1-t)+np.sin(x2-t)+np.sin(x3-t)

def gen_exact_solution(pde_name,x_dim,t_dim):
    """Generates exact solution for the heat equation for the given values of x and t."""
    # Number of points in each dimension:


    # Bounds of 'x' and 't':
    x_min, t_min = (0, 0)
    x_max, t_max = (10, 2)

    # Create tensors:
    t = np.linspace(t_min, t_max, num=t_dim).reshape(t_dim, 1)
    x = np.linspace(x_min, x_max, num=x_dim).reshape(x_dim, 1)

    if pde_name == "advection1":
        usol = np.zeros((x_dim,x_dim, t_dim)).reshape(x_dim,x_dim, t_dim)
        # Obtain the value of the exact solution for each generated point:
        for i in range(x_dim):
            for m in range(x_dim):
                for j in range(t_dim):
                    usol[i][m][j] = advection1_2D_exact_solution(x[i], x[m], t[j])

        # Save solution:
        np.savez("advection1_2d", x1=x,x2=x, t=t, usol=usol)

    if pde_name == "advection3":
        usol = np.zeros((x_dim,x_dim, t_dim)).reshape(x_dim,x_dim, t_dim)
        # Obtain the value of the exact solution for each generated point:
        for i in range(x_dim):
            for m in range(x_dim):
                for j in range(t_dim):
                    usol[i][m][j] = advection3_2D_exact_solution(x[i], x[m], t[j])

        # Save solution:
        np.savez("advection3_2d", x1=x,x2=x, t=t, usol=usol)

    if pde_name == "advection2":
        usol = np.zeros((x_dim,x_dim,x_dim, t_dim)).reshape(x_dim,x_dim,x_dim,  t_dim)
        # Obtain the value of the exact solution for each generated point:
        for i in range(x_dim):
            for j in range(x_dim):
                for m in range(x_dim):
                    for n in range(t_dim):
                        usol[i][j][m][n] = advection2_3D_exact_solution(x[i], x[j], x[m], t[n])

        # Save solution:
        np.savez("advection2_3d", x1=x, x2=x, x3=x, t=t, usol=usol)

    if pde_name == "advection4":
        usol = np.zeros((x_dim,x_dim,x_dim, t_dim)).reshape(x_dim,x_dim,x_dim,  t_dim)
        # Obtain the value of the exact solution for each generated point:
        for i in range(x_dim):
            for j in range(x_dim):
                for m in range(x_dim):
                    for n in range(t_dim):
                        usol[i][j][m][n] = advection4_3D_exact_solution(x[i], x[j], x[m], t[n])

        # Save solution:
        np.savez("advection4_3d", x1=x, x2=x, x3=x, t=t, usol=usol)



import random
import numpy as np
import math


def dataset2D_advection1(sample=10):
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
        t = random.uniform(0, 2)
        # 计算y的值
        y = x1+x2-2*t
        # 将x1、x2和y的值添加到列表中
        x1_list.append(x1)
        x2_list.append(x2)
        t_list.append(t)
        y_list.append(y)


    return [x1_list,x2_list,t_list],y_list



def dataset3D_advection2(sample=15):
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
        t = random.uniform(0, 2)
        # 计算y的值
        y = x1+x2+x3-3*t
        # 将x1、x2和y的值添加到列表中
        x1_list.append(x1)
        x2_list.append(x2)
        x3_list.append(x3)
        t_list.append(t)
        y_list.append(y)


    return [x1_list,x2_list,x3_list,t_list],y_list


def dataset2D_advection3(sample=10):
    # 创建空列表存储x1、x2和y的值
    x1_list = []
    x2_list = []
    t_list = []
    y_list = []

    # 循环n次，随机生成n个采样点
    for i in range(sample):
        # 随机生成x1和x2的值
        x1 = random.uniform(0, 10)
        x2 = random.uniform(0, 10)
        t = random.uniform(0, 2)
        # 计算y的值
        y = math.sin(x1-t)+math.sin(x2-t)
        # 将x1、x2和y的值添加到列表中
        x1_list.append(x1)
        x2_list.append(x2)
        t_list.append(t)
        y_list.append(y)


    return [x1_list,x2_list,t_list],y_list

def dataset3D_advection4(sample=15):
    # 创建空列表存储x1、x2和y的值
    x1_list = []
    x2_list = []
    x3_list = []
    t_list = []
    y_list = []

    # 循环n次，随机生成n个采样点
    for i in range(sample):
        # 随机生成x1和x2的值
        x1 = random.uniform(0, 10)
        x2 = random.uniform(0, 10)
        x3 = random.uniform(0, 10)
        t = random.uniform(0, 2)
        # 计算y的值
        y = math.sin(x1-t)+math.sin(x2-t)+math.sin(x3-t)
        # 将x1、x2和y的值添加到列表中
        x1_list.append(x1)
        x2_list.append(x2)
        x3_list.append(x3)
        t_list.append(t)
        y_list.append(y)


    return [x1_list,x2_list,x3_list,t_list],y_list
if __name__ == '__main__':
    # x_dim = 20
    # t_dim = 20
    # pde_name = "advection4"
    # gen_exact_solution(pde_name,x_dim,t_dim)
    c=np.load('advection3_2d.npz')
    print(c['x1'])

