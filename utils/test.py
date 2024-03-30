from sympy import *
init_printing()

x, y,t, k = symbols('x y t k')
u = 1 / (4 * pi * k * t)* exp((-x**2-y**2) / (4 * k * t))
res = diff(u, x)
res1 = diff(res, x)
print(res,res1)

resy = diff(u, y)
res1y = diff(resy, y)
print(resy,res1y)


res2 = diff(u, t)
if expand(k*(res1+res1y)) == expand(res2):
    print("True")
print(expand(k*(res1+res1y))-expand(res2))
pow(pow(add(mul(add(x1, x2), pow(x1, 1)), mul(add(x1, x2), add(x1, t))), add(mul(exp(x1), exp(1)), mul(mul(1, 1), exp(t)))), mul(pow(pow(1, 1), 0.0009491776671769307), 9.999999985098839))