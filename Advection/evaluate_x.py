

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

def evalSymbAD(individual,toolbox,X_train,y,d):
    expr = toolbox.compile(expr=individual)

    if d == 2:
        try:
            y_pred = [expr(X_train[0][i], X_train[1][i], X_train[2][i]) for i in range(len(X_train[0]))]
            error = [(y_pred[i] - y[i]) ** 2 for i in range(len(X_train[0]))]
        except ValueError:
            print(individual)
        mse = math.fsum(error) / len(error)
        X_pde_train1 = torch.linspace(0, 1, 100 + 1)
        X_pde_train1 = X_pde_train1.clone().detach().requires_grad_(True)
        X_pde_train2 = torch.linspace(0, 1, 100 + 1)
        X_pde_train2 = X_pde_train2.clone().detach().requires_grad_(True)
        X_pde_train3 = torch.linspace(0, 2, 100 + 1)
        X_pde_train3 = X_pde_train3.clone().detach().requires_grad_(True)
        pde_error1 = torch_Advection2d(expr, X_pde_train1,X_pde_train2,X_pde_train3)
        pde_error2 = initial_Advection2d(expr, X_pde_train1,X_pde_train2)


    elif d == 3:
        try:
            y_pred = [expr(X_train[0][i], X_train[1][i], X_train[2][i], X_train[3][i]) for i in range(len(X_train[0]))]
            error = [(y_pred[i] - y[i]) ** 2 for i in range(len(X_train[0]))]
        except ValueError:
            print(individual)
        X_pde_train1 = torch.linspace(0, 1, 100 + 1)
        X_pde_train1 = X_pde_train1.clone().detach().requires_grad_(True)
        X_pde_train2 = torch.linspace(0, 1, 100 + 1)
        X_pde_train2 = X_pde_train2.clone().detach().requires_grad_(True)
        X_pde_train3 = torch.linspace(0, 1, 100 + 1)
        X_pde_train3 = X_pde_train3.clone().detach().requires_grad_(True)
        X_pde_train4 = torch.linspace(0, 2, 100 + 1)
        X_pde_train4 = X_pde_train4.clone().detach().requires_grad_(True)
        pde_error1 = torch_Advection3d(expr, X_pde_train1, X_pde_train2, X_pde_train3,X_pde_train4)
        pde_error2 = initial_Advection3d(expr, X_pde_train1, X_pde_train2,X_pde_train3,)

        mse = math.fsum(error) / len(error)
    if mse<0.01:
        mse = 0
    return [pde_error1+pde_error2+mse]

def torch_Advection2d(expr, X1_train, X2_train,t_train,):

    x1, x2, t = Symbol("x1"), Symbol("x2"), Symbol("t")
    expr = expr(x1, x2,t)
    simplified_expr = expand(expr)
    f_torch = sympytorch.SymPyModule(expressions=[simplified_expr])
    try:
        f_x1_x2_t = f_torch(x1=X1_train, x2=X2_train,t=t_train)
        f_x1_x2_t = f_x1_x2_t.reshape(X1_train.shape)
    except:
        return 100




    # 一阶导
    try:
        du_dt = torch.autograd.grad(f_x1_x2_t.sum(), t_train, create_graph=True)
        if du_dt[0] is None:
            du_dt = torch.zeros(X1_train.size(), dtype=X1_train.dtype)
        else:
            du_dt = du_dt[0]
    except:
        du_dt = torch.zeros(X1_train.size(), dtype=X1_train.dtype)



    # 一阶导
    try:
        du_dx1 = torch.autograd.grad(f_x1_x2_t.sum(), X1_train, allow_unused=True)
        if du_dx1[0] is None:
            du_dx1 = torch.zeros(X1_train.size(), dtype=X1_train.dtype)
        else:
            du_dx1 = du_dx1[0]
    except:
        du_dx1 = torch.zeros(X1_train.size(), dtype=X1_train.dtype)


    # 一阶导
    try:
        du_dx2 = torch.autograd.grad(f_x1_x2_t.sum(), X2_train, allow_unused=True)
        if du_dx2[0] is None:
            du_dx2 = torch.zeros(X2_train.size(), dtype=X2_train.dtype)
        else:
            du_dx2 = du_dx2[0]
    except:
        du_dx2 = torch.zeros(X2_train.size(), dtype=X2_train.dtype)




    du_t_x1_x2 = du_dt+du_dx1+du_dx2

    # Compute the MSE of the Heat equation

    mse = F.mse_loss(du_t_x1_x2, torch.zeros_like(du_t_x1_x2))
    return mse.item()

def initial_Advection2d(expr, X1_train, X2_train):
    t_train = torch.zeros(X1_train.size(), dtype=X1_train.dtype)
    x1, x2, t = Symbol("x1"), Symbol("x2"), Symbol("t")
    expr = expr(x1, x2, t)
    simplified_expr = expand(expr)
    f_torch = sympytorch.SymPyModule(expressions=[simplified_expr])
    try:
        f_x1_x2_t = f_torch(x1=X1_train, x2=X2_train, t=t_train)
    except:
        f_x1_x2_t = torch.zeros(X1_train.size(), dtype=X1_train.dtype)
    du_t_x1_x2 = f_x1_x2_t.resize(len(f_x1_x2_t)) - X1_train - X2_train

    mse = F.mse_loss(du_t_x1_x2, torch.zeros_like(du_t_x1_x2))
    return mse.item()





def torch_Advection3d(expr, X1_train, X2_train, X3_train,t_train):
    x1, x2, x3, t = Symbol("x1"), Symbol("x2"), Symbol("x3"), Symbol("t")
    expr = expr(x1, x2, x3, t)
    simplified_expr = expand(expr)
    f_torch = sympytorch.SymPyModule(expressions=[simplified_expr])

    try:
        f_x1_x2_x3_t = f_torch(x1=X1_train, x2=X2_train, x3=X3_train, t=t_train)
        f_x1_x2_x3_t = f_x1_x2_x3_t.reshape(X1_train.shape)
    except:
        return 100


    # 一阶导
    try:
        du_dt = torch.autograd.grad(f_x1_x2_x3_t.sum(), t_train, create_graph=True)
        if du_dt[0] is None:
            du_dt = torch.zeros(X1_train.size(), dtype=X1_train.dtype)
        else:
            du_dt = du_dt[0]
    except:
        du_dt = torch.zeros(X1_train.size(), dtype=X1_train.dtype)


    # 一阶导
    try:
        du_dx1 = torch.autograd.grad(f_x1_x2_x3_t.sum(), X1_train,  allow_unused=True)
        if du_dx1[0] is None:
            du_dx1 = torch.zeros(X1_train.size(), dtype=X1_train.dtype)
        else:
            du_dx1 = du_dx1[0]
    except:
        du_dx1 = torch.zeros(X1_train.size(), dtype=X1_train.dtype)

    # 一阶导
    try:
        du_dx2 = torch.autograd.grad(f_x1_x2_x3_t.sum(), X2_train, allow_unused=True)
        if du_dx2[0] is None:
            du_dx2 = torch.zeros(X1_train.size(), dtype=X1_train.dtype)
        else:
            du_dx2 = du_dx2[0]
    except:
        du_dx2 = torch.zeros(X1_train.size(), dtype=X1_train.dtype)

    # 一阶导
    try:
        du_dx3 = torch.autograd.grad(f_x1_x2_x3_t.sum(), X3_train, allow_unused=True)
        if du_dx3 [0] is None:
            du_dx3  = torch.zeros(X1_train.size(), dtype=X1_train.dtype)
        else:
            du_dx3  = du_dx3[0]
    except:
        du_dx3 = torch.zeros(X1_train.size(), dtype=X1_train.dtype)

    du_x1_x2_x3 = du_dt + du_dx1+du_dx2+du_dx3

    # Compute the MSE of the Poisson equation
    mse = F.mse_loss(du_x1_x2_x3, torch.zeros_like(du_x1_x2_x3))
    # print(mse.item())
    return mse.item()



def initial_Advection3d(expr, X1_train, X2_train, X3_train):
    x1, x2, x3, t = Symbol("x1"), Symbol("x2"), Symbol("x3"), Symbol("t")
    expr = expr(x1, x2, x3, t)
    simplified_expr = expand(expr)
    f_torch = sympytorch.SymPyModule(expressions=[simplified_expr])

    try:
        f_x1_x2_x3_t = f_torch(x1=X1_train, x2=X2_train, x3=X3_train, t=torch.zeros(X1_train.size(), dtype=X1_train.dtype))
    except:
        return 10



    du_t_x1_x2 = f_x1_x2_x3_t.resize(len(f_x1_x2_x3_t)) - X1_train - X2_train- X3_train

    mse = F.mse_loss(du_t_x1_x2, torch.zeros_like(du_t_x1_x2))
    return mse.item()

