# coding:UTF-8
# @Time: 2023/8/14 12:03
# @Author: Lulu Cao
# @File: evaluate_x.py
# @Software: PyCharm

# coding:UTF-8
# @Time: 2023/5/29 9:31
# @Author: Lulu Cao
# @File: evaluate_x.py
# @Software: PyCharm

# coding:UTF-8
# @Time: 2023/4/23 10:10
# @Author: Lulu Cao
# @File: evaluate_x.py
# @Software: PyCharm

import math
from sympy import symbols, diff, lambdify,Derivative,Function,simplify,expand
import warnings
import numpy as np
import sympy
import torch
import torch.nn.functional as F
from sympy import Symbol
from sympy.utilities.lambdify import lambdify
import sympy, torch, sympytorch
import inspect
device_ids = [0]
device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else 'cpu')

def evalSymbAD(individual,toolbox,X_train,y,d,k=0.4):
    expr = toolbox.compile(expr=individual)

    if d == 2:
        try:
            y_pred = [expr(X_train[0][i], X_train[1][i], X_train[2][i]) for i in range(len(X_train[0]))]
            error = [(y_pred[i] - y[i]) ** 2 for i in range(len(X_train[0]))]
            mse = math.fsum(error) / len(error)
        except:
            return [1000]
            #print(individual)

        X_pde_train1 = torch.linspace(0, 1, 100 + 1)
        X_pde_train1 = X_pde_train1.clone().detach().requires_grad_(True)
        X_pde_train2 = torch.linspace(0, 1, 100 + 1)
        X_pde_train2 = X_pde_train2.clone().detach().requires_grad_(True)
        X_pde_train3 = torch.linspace(0.1, 1, 100 + 1)
        X_pde_train3 = X_pde_train3.clone().detach().requires_grad_(True)
        pde_error1 = torch_Heat2d(expr, X_pde_train1,X_pde_train2,X_pde_train3,k=k)


    elif d == 3:
        try:
            y_pred = [expr(X_train[0][i], X_train[1][i], X_train[2][i], X_train[3][i]) for i in range(len(X_train[0]))]
            error = [(y_pred[i] - y[i]) ** 2 for i in range(len(X_train[0]))]
            mse = math.fsum(error) / len(error)
        except:
            #print(individual)
            return [1000]
        X_pde_train1 = torch.linspace(0, 0.5, 100 + 1)
        X_pde_train1 = X_pde_train1.clone().detach().requires_grad_(True)
        X_pde_train2 = torch.linspace(0.5, 1, 100 + 1)
        X_pde_train2 = X_pde_train2.clone().detach().requires_grad_(True)
        X_pde_train3 = torch.linspace(0, 1, 100 + 1)
        X_pde_train3 = X_pde_train3.clone().detach().requires_grad_(True)
        X_pde_train4 = torch.linspace(0.1, 1, 100 + 1)
        X_pde_train4 = X_pde_train4.clone().detach().requires_grad_(True)
        pde_error1 = torch_Heat3d(expr, X_pde_train1, X_pde_train2, X_pde_train3,X_pde_train4,k=k)

    if mse<0.01:
        mse = 0
    return [pde_error1+mse]

def torch_Heat2d(expr, X1_train, X2_train,t_train,k=0.4):

    x1, x2, t = Symbol("x1"), Symbol("x2"), Symbol("t")

    expr = expr(x1, x2,t)
    simplified_expr = expand(expr)
    f_torch = sympytorch.SymPyModule(expressions=[simplified_expr])
    try:
        f_x1_x2_t = f_torch(x1=X1_train, x2=X2_train,t=t_train)
        f_x1_x2_t = f_x1_x2_t.reshape(X1_train.shape)
    except:
        return 10

    try:
        # 一阶导
        du_dt = torch.autograd.grad(f_x1_x2_t.sum(), t_train, create_graph=True)
        if du_dt[0] is None:
            du_dt = torch.zeros(X1_train.size(), dtype=X1_train.dtype)
        else:
            du_dt = du_dt[0]
    except:
        du_dt = torch.zeros(X1_train.size(), dtype=X1_train.dtype)

    try:
        f_x1_x2_t = f_x1_x2_t.reshape(X1_train.shape)
        # 一阶导
        du2_dx1 = torch.autograd.grad(f_x1_x2_t.sum(), X1_train, create_graph=True)
        if du2_dx1[0] is None:
            du2_dx1 = torch.zeros(X1_train.size(), dtype=X1_train.dtype)
        else:
            # 二阶导
            du2_dx1 = torch.autograd.grad(du2_dx1[0].sum(), X1_train, allow_unused=True)

        if du2_dx1[0] is None:
            du2_dx1 = torch.zeros(X1_train.size(), dtype=X1_train.dtype)
        else:
            du2_dx1= du2_dx1[0]

        # 一阶导
        du2_dx2 = torch.autograd.grad(f_x1_x2_t.sum(), X2_train, create_graph=True)
        if du2_dx2[0] is None:
            du2_dx2 = torch.zeros(X2_train.size(), dtype=X2_train.dtype)
        else:
            # 二阶导
            du2_dx2 = torch.autograd.grad(du2_dx2[0].sum(), X2_train, allow_unused=True)
        if du2_dx2[0] is None:
            du2_dx2 = torch.zeros(X1_train.size(), dtype=X1_train.dtype)
        else:
            du2_dx2= du2_dx2[0]

        du_x1_x2 = du2_dx1+du2_dx2
    except:
        du_x1_x2 = torch.zeros(X1_train.size(), dtype=X1_train.dtype)
    #print("The grad fn for a is", du_dt,du_x1_x2)
    du_x1_x2 = du_dt-k*du_x1_x2

    # Compute the MSE of the Heat equation

    mse = F.mse_loss(du_x1_x2, torch.zeros_like(du_x1_x2))
    return mse.item()








def torch_Heat3d(expr, X1_train, X2_train, X3_train,t_train,k=0.4):
    x1, x2, x3, t = Symbol("x1"), Symbol("x2"), Symbol("x3"), Symbol("t")
    expr = expr(x1, x2, x3, t)
    simplified_expr = expand(expr)
    f_torch = sympytorch.SymPyModule(expressions=[simplified_expr])
    try:
        f_x1_x2_x3_t = f_torch(x1=X1_train, x2=X2_train, x3=X3_train, t=t_train)
        f_x1_x2_x3_t = f_x1_x2_x3_t.reshape(X1_train.shape)
    except:
        return 10

    try:
        # 一阶导
        du_dt = torch.autograd.grad(f_x1_x2_x3_t.sum(), t_train, create_graph=True)
        if du_dt[0] is None:
            du_dt = torch.zeros(X1_train.size(), dtype=X1_train.dtype)
        else:
            du_dt = du_dt[0]
    except:
        du_dt = torch.zeros(X1_train.size(), dtype=X1_train.dtype)


    try:
        f_x1_x2_x3_t = f_x1_x2_x3_t.reshape(X1_train.shape)
        # First derivative
        du2_dx1 = torch.autograd.grad(f_x1_x2_x3_t.sum(), X1_train, create_graph=True)
        if du2_dx1[0] is None:
            du2_dx1 = torch.zeros(X1_train.size(), dtype=X1_train.dtype)
        else:
            # Second derivative
            du2_dx1 = torch.autograd.grad(du2_dx1[0].sum(), X1_train, allow_unused=True)

        if du2_dx1[0] is None:
            du2_dx1 = torch.zeros(X1_train.size(), dtype=X1_train.dtype)
        else:
            du2_dx1= du2_dx1[0]

        # First derivative
        du2_dx2 = torch.autograd.grad(f_x1_x2_x3_t.sum(), X2_train, create_graph=True)
        if du2_dx2[0] is None:
            du2_dx2 = torch.zeros(X2_train.size(), dtype=X2_train.dtype)
        else:
            # Second derivative
            du2_dx2 = torch.autograd.grad(du2_dx2[0].sum(), X2_train, allow_unused=True)
        if du2_dx2[0] is None:
            du2_dx2 = torch.zeros(X1_train.size(), dtype=X1_train.dtype)
        else:
            du2_dx2= du2_dx2[0]

        # First derivative
        du2_dx3 = torch.autograd.grad(f_x1_x2_x3_t.sum(), X3_train, create_graph=True)
        if du2_dx3[0] is None:
            du2_dx3 = torch.zeros(X3_train.size(), dtype=X3_train.dtype)
        else:
            # Second derivative
            du2_dx3 = torch.autograd.grad(du2_dx3[0].sum(), X3_train, allow_unused=True)
        if du2_dx3[0] is None:
            du2_dx3 = torch.zeros(X3_train.size(), dtype=X3_train.dtype)
        else:
            du2_dx3= du2_dx3[0]

        du_x1_x2_x3 = du2_dx1 + du2_dx2 + du2_dx3
    except:
        du_x1_x2_x3 = torch.zeros(X1_train.size(), dtype=X1_train.dtype)
    du_x1_x2_x3 = du_dt - k * du_x1_x2_x3

    # Compute the MSE of the Poisson equation
    mse = F.mse_loss(du_x1_x2_x3, torch.zeros_like(du_x1_x2_x3))
    # print(mse.item())
    return mse.item()



