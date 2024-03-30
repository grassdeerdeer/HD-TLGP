# coding:UTF-8
# @Time: 2023/8/15 17:37
# @Author: Lulu Cao
# @File: get_exact_advection.py
# @Software: PyCharm
import numpy as np
from sympy.parsing.sympy_parser import parse_expr
from sklearn.metrics import mean_squared_error
import numpy as np
import sympy
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

if __name__ == '__main__':
    # advection1

    adv1 = "-1.999*t + x1 + x2"
    expr = parse_expr(adv1, evaluate=True)
    testdata = np.load("advection1_2d.npz")

    x1, x2, t = sympy.symbols("x1,x2,t")
    expr = sympy.lambdify([x1,x2,t], expr, "numpy")

    x11,x22,tt,usol = testdata['x1'],testdata['x2'],testdata['t'],testdata['usol']
    y_pred = []
    y_true = []
    for i in range(len(x11)):
        for m in range(len(x22)):
            for j in range(len(tt)):
                # print(x11[i],x22[m],tt[j])
                y_true.append(usol[i][m][j])
                y_pred.append(expr(x11[i],x22[m],tt[j]))
    mse = mean_squared_error(y_pred, y_true)
    print(f"MSE: {mse}")
    error = np.array(y_pred) - np.array(y_true)
    variance = np.var(error)

    print(f"MSE: {mse}")
    print(f"Variance of errors: {variance}")


