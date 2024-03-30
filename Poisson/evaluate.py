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

def evalSymbAD(individual,toolbox,X_train,y,d):
    expr = toolbox.compile(expr=individual)




    if d == 1:
        try:
            y_pred = [expr(X_train[0][i]) for i in range(len(X_train[0]))]
            error = [(y_pred[i] - y[i]) ** 2 for i in range(len(X_train[0]))]
        except ValueError:
            print(individual)
        X_pde_train = torch.linspace(0, 1, 100 + 1)
        X_pde_train = X_pde_train.clone().detach().requires_grad_(True)
        mse = math.fsum(error) / len(error)
        pde_error1 = torch_Poisson1d_1(expr, X_pde_train)
        pde_error2 = Poisson1d_2(expr, X_pde_train)



    elif d == 2:
        try:
            y_pred = [expr(X_train[0][i], X_train[1][i]) for i in range(len(X_train[0]))]
            error = [(y_pred[i] - y[i]) ** 2 for i in range(len(X_train[0]))]
        except ValueError:
            print(individual)
        mse = math.fsum(error) / len(error)
        X_pde_train1 = torch.linspace(0, 1, 100 + 1)
        X_pde_train1 = X_pde_train1.clone().detach().requires_grad_(True)
        X_pde_train2 = torch.linspace(0, 1, 100 + 1)
        X_pde_train2 = X_pde_train2.clone().detach().requires_grad_(True)
        pde_error1 = torch_Poisson2d(expr, X_pde_train1,X_pde_train2)
        pde_error2 = Poisson2d_2(expr, X_pde_train1)


    elif d == 3:
        try:
            y_pred = [expr(X_train[0][i], X_train[1][i], X_train[2][i]) for i in range(len(X_train[0]))]
            error = [(y_pred[i] - y[i]) ** 2 for i in range(len(X_train[0]))]
        except ValueError:
            print(individual)
        X_pde_train1 = torch.linspace(0, 1, 100 + 1)
        X_pde_train1 = X_pde_train1.clone().detach().requires_grad_(True)
        X_pde_train2 = torch.linspace(0, 1, 100 + 1)
        X_pde_train2 = X_pde_train2.clone().detach().requires_grad_(True)
        X_pde_train3 = torch.linspace(0, 1, 100 + 1)
        X_pde_train3 = X_pde_train3.clone().detach().requires_grad_(True)
        pde_error1 = torch_Poisson3d(expr, X_pde_train1, X_pde_train2, X_pde_train3)
        pde_error2 = Poisson3d_2(expr, X_pde_train1)

        mse = math.fsum(error) / len(error)
    if mse<0.01:
        mse = 0


    return [pde_error1+pde_error2+mse]

def torch_Poisson1d_1(expr,X_train):
    x1 = Symbol("x1")
    expr = expr(x1)
    simplified_expr = expand(expr)
    # simplified_expr  = lambda x: sympy.sin(3.14*x)
    # simplified_expr = simplified_expr(x1)
    f_torch = sympytorch.SymPyModule(expressions=[simplified_expr])
    f_x1 = f_torch(x1=X_train)
    try:
        f_x1 = f_x1.reshape(X_train.shape)
        du2_dx2 = torch.autograd.grad(f_x1.sum(), X_train, create_graph=True)
        if du2_dx2[0] is None:
            du2_dx2 = torch.zeros(X_train.size(), dtype=X_train.dtype)

        else:
            du2_dx2 = torch.autograd.grad(du2_dx2[0].sum(), X_train, allow_unused=True)

        if du2_dx2[0] is None:
            du2_dx2 = torch.zeros(X_train.size(), dtype=X_train.dtype)
        else:
            du2_dx2 = du2_dx2[0]
    except:
        du2_dx2 = torch.zeros(X_train.size(), dtype=X_train.dtype)

    # Define the source term
    source_term = -torch.tensor(np.pi) ** 2 * torch.sin(torch.tensor(np.pi) * X_train)
    du2_dx2 = du2_dx2 - source_term

    # Compute the MSE of the Poisson equation
    mse = F.mse_loss(du2_dx2 , torch.zeros_like(du2_dx2))
    #print(mse.item())
    return mse.item()

def Poisson1d_2(f,X_train):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            return abs(f(0))+abs(f(1))
        except  Warning:
            return 1000
        except ZeroDivisionError:
            return 1000



def torch_Poisson2d(expr, X1_train, X2_train):
    x1, x2 = Symbol("x1"), Symbol("x2")
    expr = expr(x1, x2)
    simplified_expr = expand(expr)
    f_torch = sympytorch.SymPyModule(expressions=[simplified_expr])
    f_x1_x2 = f_torch(x1=X1_train, x2=X2_train)

    try:
        f_x1_x2 = f_x1_x2.reshape(X1_train.shape)
        # 一阶导
        du2_dx1 = torch.autograd.grad(f_x1_x2.sum(), X1_train, create_graph=True)
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
        du2_dx2 = torch.autograd.grad(f_x1_x2.sum(), X2_train, create_graph=True)
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



# Define the source term
    source_term = -2*torch.tensor(np.pi) ** 2 * torch.sin(torch.tensor(np.pi) * X1_train) * torch.sin(torch.tensor(np.pi) * X2_train)
    du_x1_x2 = du_x1_x2 - source_term

    # Compute the MSE of the Poisson equation
    mse = F.mse_loss(du_x1_x2 , torch.zeros_like(du_x1_x2))
    #print(mse.item())
    return mse.item()

def Poisson2d_1(f,X_train):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            # 定义两个符号变量
            x1 = symbols('x1')
            x2 = sympy.Symbol('x2')

            # 定义 2DPoisson 偏微分方程
            w = Function('w')(x1,x2)

            pde1 = Derivative(w, x1, 2)+Derivative(w, x2, 2)+2*sympy.pi**2 * sympy.sin(sympy.pi * x1) * sympy.sin(sympy.pi * x2)
            # 将 lambda 函数转换为 SymPy 表达式
            w_expr = simplify(f(x1,x2))

            # 计算偏微分方程的值
            pde_value = pde1.subs(w, w_expr).doit()
            result = expand(pde_value)
            result = lambdify([x1,x2], result)
            error1 = [abs(result(X_train[i],X_train[i]))  for i in range(len(X_train))]

            return math.fsum(error1)
        except  Warning:
            return 1000


def Poisson2d_2(f,X_train):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            x1b0 = [abs(f(0,X_train[i]))  for i in range(len(X_train))]
            x1b1 = [abs(f(1, X_train[i])) for i in range(len(X_train))]

            x2b0 = [abs(f(X_train[i],0)) for i in range(len(X_train))]
            x2b1 = [abs(f(X_train[i],1)) for i in range(len(X_train))]
            return math.fsum(x1b0)+math.fsum(x1b1)+math.fsum(x2b0)+math.fsum(x2b1)
        except  Warning:
            return 1000
        except ZeroDivisionError:
            return 1000



def torch_Poisson3d(expr, X1_train, X2_train, X3_train):
    x1, x2, x3 = Symbol("x1"), Symbol("x2"), Symbol("x3")
    expr = expr(x1, x2, x3)
    simplified_expr = expand(expr)
    f_torch = sympytorch.SymPyModule(expressions=[simplified_expr])
    f_x1_x2_x3 = f_torch(x1=X1_train, x2=X2_train, x3=X3_train)

    try:
        f_x1_x2_x3 = f_x1_x2_x3.reshape(X1_train.shape)
        # First derivative
        du2_dx1 = torch.autograd.grad(f_x1_x2_x3.sum(), X1_train, create_graph=True)
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
        du2_dx2 = torch.autograd.grad(f_x1_x2_x3.sum(), X2_train, create_graph=True)
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
        du2_dx3 = torch.autograd.grad(f_x1_x2_x3.sum(), X3_train, create_graph=True)
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

    # Define the source term
    source_term = -3*torch.tensor(np.pi) ** 2 * torch.sin(torch.tensor(np.pi) * X1_train) * torch.sin(
        torch.tensor(np.pi) * X2_train)* torch.sin(
        torch.tensor(np.pi) * X3_train)
    du_x1_x2_x3 = du_x1_x2_x3 - source_term

    # Compute the MSE of the Poisson equation
    mse = F.mse_loss(du_x1_x2_x3, torch.zeros_like(du_x1_x2_x3))
    # print(mse.item())
    return mse.item()



def Poisson3d_1(f,X_train):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            # 定义两个符号变量
            x1 = symbols('x1')
            x2 = symbols('x2')
            x3 = symbols('x3')

            # 定义 2DPoisson 偏微分方程
            w = Function('w')(x1,x2,x3)

            pde1 = Derivative(w, x1, 2)+Derivative(w, x2, 2)+Derivative(w, x3, 2)+3*sympy.pi**2 * sympy.sin(sympy.pi * x1) * sympy.sin(sympy.pi * x2)* sympy.sin(sympy.pi * x3)
            # 将 lambda 函数转换为 SymPy 表达式
            w_expr = simplify(f(x1,x2,x3))

            # 计算偏微分方程的值
            pde_value = pde1.subs(w, w_expr).doit()
            result = expand(pde_value)
            result = lambdify([x1,x2,x3], result)
            error1 = [abs(result(X_train[i],X_train[i],X_train[i]))  for i in range(len(X_train))]

            return math.fsum(error1)
        except  Warning:
            return 1000


def Poisson3d_2(f,X_train):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            x1b0 = [abs(f(0,X_train[i],X_train[i]))  for i in range(len(X_train))]
            x1b1 = [abs(f(1, X_train[i],X_train[i])) for i in range(len(X_train))]

            x2b0 = [abs(f(X_train[i],0,X_train[i])) for i in range(len(X_train))]
            x2b1 = [abs(f(X_train[i],1,X_train[i])) for i in range(len(X_train))]

            x3b0 = [abs(f(X_train[i],X_train[i], 0)) for i in range(len(X_train))]
            x3b1 = [abs(f(X_train[i],X_train[i], 1)) for i in range(len(X_train))]
            return math.fsum(x1b0)+math.fsum(x1b1)+math.fsum(x2b0)+math.fsum(x2b1)+math.fsum(x3b0)+math.fsum(x3b1)
        except  Warning:
            return 1000
        except ZeroDivisionError:
            return 1000
