# coding:UTF-8
# @Time: 2023/5/29 11:05
# @Author: Lulu Cao
# @File: PoissonDataset.py
# @Software: PyCharm
import random
import numpy as np
import math


def poission1_2D_exact_solution(x1, x2,):
    """
    :param x1: np.ndarray
    :param x2: np.ndarray
    :param t: np.ndarray
    :return:
    """
    return np.sin(np.pi*x1)*np.sin(np.pi*x2)


def poission2_3D_exact_solution(x1, x2, x3):
    """
    :param x1: np.ndarray
    :param x2: np.ndarray
    :param x3: np.ndarray
    :param t: np.ndarray
    :return:
    """
    return np.sin(np.pi*x1)*np.sin(np.pi*x2)*np.sin(np.pi*x3)

def poission3_2D_exact_solution(x1, x2):
    """
    :param x1: np.ndarray
    :param x2: np.ndarray
    :param t: np.ndarray
    :return:
    """
    return 0.25-0.25*x1*x1-0.25*x2*x2


def poission4_3D_exact_solution(x1, x2, x3):
    """
    :param x1: np.ndarray
    :param x2: np.ndarray
    :param x3: np.ndarray
    :param t: np.ndarray
    :return:
    """
    return 0.167-0.167*x1*x1-0.167*x2*x2-0.167*x3*x3

def gen_exact_solution(pde_name,x_dim):
    """Generates exact solution for the heat equation for the given values of x and t."""
    # Number of points in each dimension:


    # Bounds of 'x' and 't':
    x_min,x_max = 0,1

    # Create tensors:
    x = np.linspace(x_min, x_max, num=x_dim).reshape(x_dim, 1)

    if pde_name == "poission1":
        usol = np.zeros((x_dim,x_dim)).reshape(x_dim,x_dim)
        # Obtain the value of the exact solution for each generated point:
        for i in range(x_dim):
            for j in range(x_dim):
                    usol[i][j] = poission1_2D_exact_solution(x[i], x[j])

        # Save solution:
        np.savez("poission1_2d", x1=x, x2=x, usol=usol)

    if pde_name == "poission3":
        usol = np.zeros((x_dim,x_dim, )).reshape(x_dim,x_dim,)
        # Obtain the value of the exact solution for each generated point:
        for i in range(x_dim):
            for j in range(x_dim):
                    usol[i][j] = poission3_2D_exact_solution(x[i], x[j], )

        # Save solution:
        np.savez("poission3_2d", x1=x,x2=x, usol=usol)

    if pde_name == "poission2":
        usol = np.zeros((x_dim,x_dim,x_dim,)).reshape(x_dim,x_dim,x_dim, )
        # Obtain the value of the exact solution for each generated point:
        for i in range(x_dim):
            for j in range(x_dim):
                for m in range(x_dim):
                        usol[i][j][m] = poission2_3D_exact_solution(x[i], x[j], x[m],)

        # Save solution:
        np.savez("poission2_3d", x1=x, x2=x, x3=x, usol=usol)

    if pde_name == "poission4":
        usol = np.zeros((x_dim,x_dim,x_dim, )).reshape(x_dim,x_dim,x_dim, )
        # Obtain the value of the exact solution for each generated point:
        for i in range(x_dim):
            for j in range(x_dim):
                for m in range(x_dim):
                        usol[i][j][m]= poission4_3D_exact_solution(x[i], x[j], x[m], )

        # Save solution:
        np.savez("poission4_3d", x1=x, x2=x, x3=x, usol=usol)




























def dataset1D(sample=10,x_min=0,x_max=1):
    # 准备数据
    # 生成一个符合pde关系的数据集,如果有解析解，生成符合解析解的也可以
    X_train = [random.uniform(x_min, x_max) for _ in range(sample)]
    X_train.append(0)
    X_train.append(1)
    X_train = np.array(X_train)

    y = []
    for x in X_train:
        y_temp = math.sin(math.pi * x)
        y.append(y_temp)
    return [X_train],y

def dataset2D(sample=10,x_min=0,x_max=1):
    # 创建空列表存储x1、x2和y的值
    x1_list = []
    x2_list = []
    y_list = []

    # 循环n次，随机生成n个采样点
    for i in range(sample):
        # 随机生成x1和x2的值
        x1 = random.uniform(x_min, x_max)
        x2 = random.uniform(x_min, x_max)
        # 计算y的值
        y = math.sin(math.pi * x1) * math.sin(math.pi * x2)
        # 将x1、x2和y的值添加到列表中
        x1_list.append(x1)
        x2_list.append(x2)
        y_list.append(y)


    return [x1_list,x2_list],y_list



def dataset3D(sample=10,x_min=0,x_max=1):
    # 创建空列表存储x1、x2和y的值
    x1_list = []
    x2_list = []
    x3_list = []
    y_list = []

    # 循环n次，随机生成n个采样点
    for i in range(sample):
        # 随机生成x1和x2的值
        x1 = random.uniform(x_min, x_max)
        x2 = random.uniform(x_min, x_max)
        x3 = random.uniform(x_min, x_max)
        # 计算y的值
        y = math.sin(math.pi * x1) * math.sin(math.pi * x2)* math.sin(math.pi * x3)
        # 将x1、x2和y的值添加到列表中
        x1_list.append(x1)
        x2_list.append(x2)
        x3_list.append(x3)
        y_list.append(y)


    return [x1_list,x2_list,x3_list],y_list





def dataset2D_poission3(sample=10):
    # 创建空列表存储x1、x2和y的值
    x1_list = []
    x2_list = []
    y_list = []

    # 循环n次，随机生成n个采样点
    for i in range(sample):
        # 随机生成x1和x2的值
        x1 = random.uniform(0, 1)
        x2 = random.uniform(0, 1)
        # 计算y的值
        y = 0.25-0.25*x1*x1-0.25*x2*x2
        # 将x1、x2和y的值添加到列表中
        x1_list.append(x1)
        x2_list.append(x2)
        y_list.append(y)


    return [x1_list,x2_list],y_list

def dataset3D_poission4(sample=15):
    # 创建空列表存储x1、x2和y的值
    x1_list = []
    x2_list = []
    x3_list = []
    y_list = []

    # 循环n次，随机生成n个采样点
    for i in range(sample):
        # 随机生成x1和x2的值
        x1 = random.uniform(0, 1)
        x2 = random.uniform(0, 1)
        x3 = random.uniform(0, 1)
        # 计算y的值
        y = 0.167-0.167*x1*x1-0.167*x2*x2-0.167*x3*x3
        # 将x1、x2和y的值添加到列表中
        x1_list.append(x1)
        x2_list.append(x2)
        x3_list.append(x3)
        y_list.append(y)


    return [x1_list,x2_list,x3_list],y_list

if __name__ == '__main__':
    x_dim = 100  # 20
    pde_name = "poission4"
    gen_exact_solution(pde_name, x_dim)