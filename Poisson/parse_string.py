# coding:UTF-8
# @Time: 2023/5/25 17:43
# @Author: Lulu Cao
# @File: parse_string.py
# @Software: PyCharm

from sympy import Add,Mul
import regex


# 定义一个函数，接受一个字符串作为参数，返回一个转换后的字符串


def convert_power(string):
    # 如果字符串中包含"**"，则进行转换
    if "**" in string:
        # 用正则表达式匹配所有的幂运算，得到一个匹配对象的列表
        import re
        matches = re.finditer(r"\w+\*\*\d+", string)
        # 创建一个空列表，用于存放转换后的幂运算
        new_powers = []
        # 遍历匹配对象的列表
        for match in matches:
            # 获取匹配的字符串
            power = match.group()
            # 用"**"分割字符串，得到底数和指数
            base, exponent = power.split("**")
            # 将指数转换为整数
            exponent = int(exponent)
            # 用"*"重复底数指数次，得到新的幂运算
            new_power = "*".join([base] * exponent)
            # 将新的幂运算添加到列表中
            new_powers.append(new_power)
        # 用正则表达式替换所有的幂运算为新的幂运算，得到新的字符串
        new_string = re.sub(r"\w+\*\*\d+", lambda m: new_powers.pop(0), string)
        # 返回新的字符串
        return new_string
    # 否则，直接返回原字符串
    else:
        return string

def convert_power_sin(string):
    # 如果字符串中包含"**"，则进行转换
    pattern = r"sin\((?:[^()]+|(?R))*\)\*{2}\d+"
    while "**" in string:
        # 用正则表达式匹配所有的幂运算，得到一个匹配对象的列表
        matches = regex.finditer(pattern,string)
        # 创建一个空列表，用于存放转换后的幂运算
        new_powers = []
        # 遍历匹配对象的列表
        for match in matches:
            # 获取匹配的字符串
            power = match.group()
            # 用"**"分割字符串，得到底数和指数
            base, exponent = power.rsplit("**",1)
            # 将指数转换为整数
            exponent = int(exponent)
            # 用"*"重复底数指数次，得到新的幂运算
            new_power = "*".join([base] * exponent)
            # 将新的幂运算添加到列表中
            new_powers.append(new_power)
        # 用正则表达式替换所有的幂运算为新的幂运算，得到新的字符串
        string = regex.sub(pattern, lambda m: new_powers.pop(0), string)
        # 返回新的字符串

    return string

def to_prefix(expr):
    if expr.is_Atom:
        return str(expr)
    else:
        op = expr.func.__name__
        if isinstance(expr, Add):
            if len(expr.args) == 1:
                return to_prefix(expr.args[0])
            else:
                return f'{op}({to_prefix(expr.args[-1])}, {to_prefix(Add(*expr.args[:-1]))})'
        if isinstance(expr, Mul):
            if len(expr.args) == 1:
                return to_prefix(expr.args[0])
            else:
                return f'{op}({to_prefix(expr.args[-1])}, {to_prefix(Mul(*expr.args[:-1],evaluate=False))})'
        else:
            args = ','.join(to_prefix(arg) for arg in expr.args)
            return f'{op}({args})'


