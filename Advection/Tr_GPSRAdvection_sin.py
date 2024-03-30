# coding:UTF-8
# @Time: 2023/8/14 17:31
# @Author: Lulu Cao
# @File: Tr_GPSRAdvection_sin.py
# @Software: PyCharm


from AdvectionDataset import *
import math
import copy
from deap import base, creator, gp, tools, algorithms
import operator

import random
import numpy as np
from local_optimize import *
from evaluate_sin import *
from sympy import symbols,expand
import sympy
from parse_string import *
from sympy.parsing.sympy_parser import parse_expr

def str2ind(str1,toolbox):
    """
    由字符串生成个体
    :param str1: 字符串公式
    :param toolbox: 生成个体的toolbox
    :return: 生成的个体
    """
    sol_1d = convert_power(str1)

    sol_1d = parse_expr(sol_1d, evaluate=False)

    prefix_str = to_prefix(sol_1d)
    prefix_str = prefix_str.replace("Add", "add")
    prefix_str = prefix_str.replace("Mul", "mul")
    ind_1d = toolbox.individual_from_string(prefix_str)
    return ind_1d

def define_gp(X_train,y,d,k=0.4):


    pset = gp.PrimitiveSet("MAIN", d+1) # 创建了一个名为 "MAIN" 的原语集，其中包含1个变量，名称默认为 "ARG0"
    if d == 1:
        pset.renameArguments(ARG0='x1')
        pset.renameArguments(ARG1='t')
    elif d == 2:
        pset.renameArguments(ARG0='x1')
        pset.renameArguments(ARG1='x2')
        pset.renameArguments(ARG2='t')
    elif d == 3:
        pset.renameArguments(ARG0='x1')
        pset.renameArguments(ARG1='x2')
        pset.renameArguments(ARG2='x3')
        pset.renameArguments(ARG3='t')

    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(sympy.sin, arity=1)
    pset.addTerminal(terminal=1.0)



    # 定义一个适应度类，继承自base.Fitness，并指定weights属性为负数，表示越小越好
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

    # 定义一个个体类，继承自gp.PrimitiveTree，并添加fitness属性和params属性，并添加local_optimize方法
    creator.create("Individual", gp.PrimitiveTree,
                   fitness=creator.FitnessMin,
                   params=None,
                   local_optimize=local_optimize)

    # 创建一个基类对象，并注册相关的属性和方法
    toolbox = base.Toolbox()



    # 注册适应度评估函数，使用均方误差作为评估指标，并传入数据集作为参数
    toolbox.register("evaluate", evalSymbAD, toolbox=toolbox, X_train=X_train, y=y,d=d)



    # 注册表达式生成器，使用ramped half-and-half方法，并指定最大深度为4
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_= 3 )

    # 注册个体生成器，使用表达式生成器，并将结果转换为个体类
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

    # 注册种群生成器，使用个体生成器，并指定种群大小为1000
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 它还注册了一个编译函数，用于将树表达式转换为可调用函数
    toolbox.register("compile", gp.compile, pset=pset)




    # 注册选择算子
    # toolbox.register("select", tools.selNSGA2)  # 使用NSGA-II选择算子
    toolbox.register("select", tools.selBest) # 如果你的适应度函数返回一个元组，表示多个目标值，那么tools.selBest会根据元组中的第一个值进行排序，然后选择最优的个体。


    # 注册交叉算子
    toolbox.register("leafmate", gp.cxOnePointLeafBiased, termpb=0.1)  # 叶子偏置的单点交叉算子，可以根据一个概率参数termpb来选择交换两棵树的内部节点或叶子节点。

    # 随机选择两棵树的一个内部节点，并交换它们的子树。使用静态限制装饰器，对交叉或变异算子添加一个限制条件，例如树的高度
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=2))

    # 注册变异算子，然后对树中的每个节点，以一定的概率用 expr 生成的新表达式替换它。
    toolbox.register("expr_mut", gp.genFull, min_=1, max_=2)  # 使用完全生成法作为变异算子的基础
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)  # 使用均匀变异算子


    def from_string(expr_str, pset):
        tree = gp.PrimitiveTree.from_string(expr_str, pset)
        return tools.initIterate(creator.Individual, lambda: tree)

    toolbox.register("individual_from_string", from_string, pset=pset)


    # 定义一个个体类，继承自gp.PrimitiveTree，并添加fitness属性和params属性，并添加local_optimize方法
    def from_list(expr_list):
        tree = gp.PrimitiveTree(expr_list)
        return tools.initIterate(creator.Individual, lambda: tree);

    toolbox.register("individual_list", from_list)

    return toolbox,pset



def vac_operator2d(offspring,vacb,pset,toolbox):
    for ivac, vacind in enumerate(offspring):
        if random.random() < vacb:
            expr = gp.compile(vacind, pset)
            x1 = symbols('x1')
            x2 = symbols('x2')
            t = symbols('t')
            expr = expr(x1,x2,t)
            simplified_expr = expand(expr)
            simplified_expr = str(simplified_expr)
            simplified_expr = convert_power(simplified_expr)
            simplified_expr = convert_power_sin(simplified_expr)
            # print(simplified_expr)
            try:
                simplified_expr = parse_expr(simplified_expr, evaluate=False)
            except:
                continue
            prefix_str = to_prefix(simplified_expr)
            prefix_str = prefix_str.replace("Add", "add")
            prefix_str = prefix_str.replace("Mul", "mul")
            ind = toolbox.individual_from_string(prefix_str)

            params = [node.value for node in ind if
                      (isinstance(node, gp.Terminal) and not isinstance(node.value, str))]
            if len(params) == 0:
                continue
            index = random.randint(0, len(params) - 1)  # 随机生成一个列表索引
            params[index] = 0  # 将该索引对应的元素设为0
            i = 0
            for node in ind:
                # 检查节点是否为常数
                # if isinstance(node, gp.RAND) or (isinstance(node, gp.Terminal) and not isinstance(node.value, str)):
                if (isinstance(node, gp.Terminal) and not isinstance(node.value, str)):
                    # 修改常数值
                    node.value = params[i]
                    i += 1

            expr = gp.compile(ind, pset)
            x1 = symbols('x1')
            x2 = symbols('x2')
            t = symbols('t')
            expr = expr(x1,x2,t)
            simplified_expr = expand(expr)
            simplified_expr = str(simplified_expr)
            simplified_expr = convert_power(simplified_expr)
            simplified_expr = convert_power_sin(simplified_expr)
            simplified_expr = parse_expr(simplified_expr, evaluate=False)
            prefix_str = to_prefix(simplified_expr)
            prefix_str = prefix_str.replace("Add", "add")
            prefix_str = prefix_str.replace("Mul", "mul")
            ind = toolbox.individual_from_string(prefix_str)
            del ind.fitness.values
            offspring[ivac] = ind
    return offspring


def vac_operator3d(offspring,vacb,pset,toolbox):
    for ivac, vacind in enumerate(offspring):
        if random.random() < vacb:
            expr = gp.compile(vacind, pset)
            x1 = symbols('x1')
            x2 = symbols('x2')
            x3 = symbols('x3')
            t = symbols('t')
            expr = expr(x1,x2,x3,t)
            simplified_expr = expand(expr)
            simplified_expr = str(simplified_expr)
            simplified_expr = convert_power(simplified_expr)
            simplified_expr = convert_power_sin(simplified_expr)
            # print(simplified_expr)
            try:
                simplified_expr = parse_expr(simplified_expr, evaluate=False)
            except:
                continue
            prefix_str = to_prefix(simplified_expr)
            prefix_str = prefix_str.replace("Add", "add")
            prefix_str = prefix_str.replace("Mul", "mul")
            ind = toolbox.individual_from_string(prefix_str)

            params = [node.value for node in ind if
                      (isinstance(node, gp.Terminal) and not isinstance(node.value, str))]
            if len(params) == 0:
                continue
            index = random.randint(0, len(params) - 1)  # 随机生成一个列表索引
            params[index] = 0  # 将该索引对应的元素设为0
            i = 0
            for node in ind:
                # 检查节点是否为常数
                # if isinstance(node, gp.RAND) or (isinstance(node, gp.Terminal) and not isinstance(node.value, str)):
                if (isinstance(node, gp.Terminal) and not isinstance(node.value, str)):
                    # 修改常数值
                    node.value = params[i]
                    i += 1

            expr = gp.compile(ind, pset)
            x1 = symbols('x1')
            x2 = symbols('x2')
            x3 = symbols('x3')
            t = symbols('t')
            expr = expr(x1, x2, x3, t)
            simplified_expr = expand(expr)
            simplified_expr = str(simplified_expr)
            simplified_expr = convert_power(simplified_expr)
            simplified_expr = convert_power_sin(simplified_expr)
            simplified_expr = parse_expr(simplified_expr, evaluate=False)
            prefix_str = to_prefix(simplified_expr)
            prefix_str = prefix_str.replace("Add", "add")
            prefix_str = prefix_str.replace("Mul", "mul")
            ind = toolbox.individual_from_string(prefix_str)
            del ind.fitness.values
            offspring[ivac] = ind
    return offspring





def tr_gp(knowledge_base,toolbox,pset,X_train,y):
    # 定义进化的参数，比如种群大小、进化代数、交叉概率、变异概率等
    popnum = len(knowledge_base)*2
    ngen = 100
    cxpb = 0.6
    mutpb = 0.6
    trpb = 0.6
    prunpb = 0.6
    vacb = 0.6

    # 创建一个种群，并初始化每个个体的参数和适应度值
    pop1 = [toolbox.individual_list(copy.deepcopy(ind[:])) for ind in knowledge_base]
    pop2 = toolbox.population(n=popnum - len(pop1))
    for i, ind in enumerate(pop2):
        root_index = random.randrange(len(ind))
        subtree_slice = ind.searchSubtree(root_index)
        kb_index = random.randrange(len(knowledge_base))
        ind[subtree_slice] = copy.deepcopy(knowledge_base[kb_index])
    pop = pop1 + pop2

    best_inds = []  # 记录每一代的最优个体

    for g in range(ngen):

        new_pop = toolbox.population(n=popnum)
        new_pop[:] = list(map(toolbox.clone, best_inds)) + list(map(toolbox.clone, pop)) + list(map(toolbox.clone, new_pop))


        for i, ind in enumerate(new_pop):
            ind.local_optimize(X_train, y, pset)
            # 评估个体的适应度值
            ind.fitness.values = toolbox.evaluate(ind)
            new_pop[i] = ind


        # 选择下一代的个体
        offspring0 = toolbox.select(new_pop, popnum)
        # 克隆每个个体，避免修改原始种群
        offspring = copy.deepcopy(list(map(toolbox.clone, offspring0)))

        # 对选出的个体进行交叉和变异
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                if len(mutant)<=2:
                    continue
                toolbox.mutate(mutant)
                del mutant.fitness.values




        for ind in offspring:
            if random.random() < trpb:
                root_index = random.randrange(len(ind))
                subtree_slice = ind.searchSubtree(root_index)
                kb_index = random.randrange(len(knowledge_base))
                ind[subtree_slice] = knowledge_base[kb_index]
                del ind.fitness.values

        temple_terminal = toolbox.individual_from_string('1.0')
        for i, ind in enumerate(offspring):
            if len(ind)>100 or random.random() < prunpb:
                root_index = random.randrange(0, len(ind))
                subtree_slice = ind.searchSubtree(root_index)
                ind = ind[:subtree_slice.start] + temple_terminal[:] + ind[subtree_slice.stop:]
                offspring[i] = toolbox.individual_list(ind)

        # 化简剪枝算子
        if len(X_train) == 3: # 2d
            offspring = vac_operator2d(offspring, vacb, pset, toolbox)
        elif len(X_train) == 4:
            offspring = vac_operator3d(offspring,vacb,pset,toolbox)

        # 对每个新生成的个体进行局部优化和评估
        for i, ind in enumerate(offspring):
            if not ind.fitness.valid:
                # 编译个体的表达式为一个可执行的函数，并赋值给expr属性
                ind.local_optimize(X_train, y, pset)
                # 评估个体的适应度值
                ind.fitness.values = toolbox.evaluate(ind)
                offspring[i] = ind

        # 更新种群
        pop[:] = new_pop + offspring
        fitness_set = set()  # 创建一个空集合
        new_pops = []  # 创建一个新的种群列表
        for ind in pop:  # 遍历种群中的每个个体
            if ind.fitness not in fitness_set:  # 如果个体的适应度值不在集合中
                fitness_set.add(ind.fitness)  # 将适应度值添加到集合中
                new_pops.append(ind)  # 将个体添加到新的种群列表中
        pop = new_pops  # 用新的种群列表替换原来的种群列表
        # offspring = toolbox.selBest(pop, popnum)
        offspring = toolbox.select(pop, popnum)
        pop[:] = offspring

        # 保存最优个体
        best_ind = min(pop, key=lambda ind: ind.fitness.values[0])

        if len(X_train) == 3:
            expr = gp.compile(best_ind, pset)
            x1 = symbols('x1')
            x2 = symbols('x2')
            t = symbols('t')
            expr = expr(x1, x2, t)
            simplified_expr = expand(expr)
            simplified_expr = str(simplified_expr)
            simplified_expr = convert_power(simplified_expr)
            simplified_expr = convert_power_sin(simplified_expr)



            # print(simplified_expr)
            try:
                simplified_expr = parse_expr(simplified_expr, evaluate=False)
            except:
                continue

            prefix_str = to_prefix(simplified_expr)
            prefix_str = prefix_str.replace("Add", "add")
            prefix_str = prefix_str.replace("Mul", "mul")
            ind = toolbox.individual_from_string(prefix_str)


        elif len(X_train) == 4:
            expr = gp.compile(best_ind, pset)
            x1 = symbols('x1')
            x2 = symbols('x2')
            x3 = symbols('x3')
            t = symbols('t')
            expr = expr(x1, x2, x3, t)
            simplified_expr = expand(expr)
            simplified_expr = str(simplified_expr)
            simplified_expr = convert_power(simplified_expr)
            simplified_expr = convert_power_sin(simplified_expr)
            # print(simplified_expr)
            try:
                simplified_expr = parse_expr(simplified_expr, evaluate=False)
            except:
                continue
            prefix_str = to_prefix(simplified_expr)
            prefix_str = prefix_str.replace("Add", "add")
            prefix_str = prefix_str.replace("Mul", "mul")
            best_ind = toolbox.individual_from_string(prefix_str)
            pass

        #best_ind.local_optimize(X_train, y, pset)
        # 评估个体的适应度值
        best_ind.fitness.values = toolbox.evaluate(ind)
        best_inds.append(copy.deepcopy(best_ind))
        expr = gp.compile(best_ind, pset)


        if len(X_train) == 3:
            x1 = symbols('x1')
            x2 = symbols('x2')
            t = symbols('t')
            expr = expr(x1, x2,t)
        elif len(X_train) == 4:
            x1 = symbols('x1')
            x2 = symbols('x2')
            x3 = symbols('x3')
            t = symbols('t')
            expr = expr(x1, x2, x3,t)

        simplified_expr = expand(expr)

        # 记录当前种群的统计信息
        # record = mstats.compile(pop)
        # logbook.record(gen=g, **record)
        print(f"Generation {g}: {str(simplified_expr)}")
        if best_ind.fitness.values[0] < 1e-5:
            break

def main(d = 2):

    if d == 2:
        X_train, y_train = dataset2D_advection3(sample=20)

    elif d == 3:
        X_train, y_train = dataset3D_advection4(sample=30)

    # 定义变量
    toolbox, pset = define_gp(X_train, y_train, d)
    knowledge_base = []


    ## TODO 1: 基于1d问题的解，建立知识库

    sol_1d = f'sin(x1-t)'
    best_ind1 = str2ind(sol_1d, toolbox)



    if d==2:
        sol_2d = sol_1d.replace('x1','x2')
        best_ind2 = str2ind(sol_2d, toolbox)
        for i in range(len(best_ind1)):
            subtree_slice1 = best_ind1.searchSubtree(i)
            subtree_slice2 = best_ind2.searchSubtree(i)
            if len(best_ind1[subtree_slice1]) >= len(best_ind1):
                knowledge_base.append(toolbox.individual_list(copy.deepcopy(best_ind1[subtree_slice1])))
                knowledge_base.append(toolbox.individual_list(copy.deepcopy(best_ind2[subtree_slice2])))
                #Concatenate the two individuals
                new_expr = toolbox.individual_list(
                    copy.deepcopy(toolbox.individual_from_string('add')) + copy.deepcopy(
                        toolbox.individual_list(best_ind1[subtree_slice1])[
                        :]) + copy.deepcopy(toolbox.individual_list(best_ind2[subtree_slice2])[:]))

                knowledge_base.append(toolbox.clone(new_expr))



    if d==3:

        sol_2d = sol_1d.replace('x1','x3')
        best_ind2 = str2ind(sol_2d, toolbox)

        sol_2d = sol_1d.replace('x1', 'x2')
        best_ind3 = str2ind(sol_2d, toolbox)
        for i in range(len(best_ind1)):
            subtree_slice1 = best_ind1.searchSubtree(i)
            subtree_slice2 = best_ind2.searchSubtree(i)
            subtree_slice3 = best_ind3.searchSubtree(i)
            if len(best_ind1[subtree_slice1]) >= len(best_ind1):
                knowledge_base.append(toolbox.individual_list(copy.deepcopy(best_ind1[subtree_slice1])))
                knowledge_base.append(toolbox.individual_list(copy.deepcopy(best_ind2[subtree_slice2])))
                knowledge_base.append(toolbox.individual_list(copy.deepcopy(best_ind3[subtree_slice3])))
                #Concatenate the two individuals
                new_expr = toolbox.individual_list(toolbox.individual_from_string('add') + copy.deepcopy(
                    toolbox.individual_list(best_ind1[subtree_slice1])[:]) + copy.deepcopy(
                    toolbox.individual_list(best_ind2[subtree_slice2])[:]))
                new_expr = toolbox.individual_list(toolbox.individual_from_string('add') + copy.deepcopy(new_expr[:]) +
                                                   copy.deepcopy(toolbox.individual_list(best_ind3[subtree_slice3])[:]))
                knowledge_base.append(toolbox.clone(new_expr))

    #Leaf
    for i in range(len(knowledge_base)):
        ind1, ind2 = random.sample(knowledge_base, 2)
        child = toolbox.clone(ind1)
        child = child[:-1] + toolbox.clone(ind2[:])
        knowledge_base.append(toolbox.individual_list(child))

    pop = tr_gp(knowledge_base, toolbox, pset, X_train, y_train)





if __name__ == "__main__":
    main(d=3)
